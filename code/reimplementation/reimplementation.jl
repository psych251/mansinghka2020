using Printf, DataFrames, CSV, Logging
using DataStructures: OrderedDict
using Julog, PDDL, Gen, Plinf

using GenParticleFilters

include("utils.jl")
include("updates.jl")
include("kernels.jl")

"Online inference over a world model using a particle filter."
function world_particle_filter2(
        world_init::WorldInit, world_config::WorldConfig,
        obs_traj::Vector{State}, obs_terms::Vector{<:Term}, n_particles::Int;
        batch_size::Int=1, delay::Int=0, strata=nothing, callback=nothing,
        ess_threshold::Float64=1/4, update_proposal=nothing,
        priority_fn=w->w*0.75, resample=true, rejuvenate=nothing)
    # Construct choicemaps from observed trajectory
    @unpack domain = world_config
    n_obs = length(obs_traj)
    obs_choices = traj_choicemaps(obs_traj, domain, obs_terms;
                                  batch_size=batch_size, offset=delay)
    # Initialize particle filter
    world_args = (world_init, world_config)
    argdiffs = (UnknownChange(), NoChange(), NoChange())
    pf_state =  initialize_pf_stratified(world_model, (0, world_args...),
                                         choicemap(), strata, n_particles)
    # Run callback for initial states if delay is used
    if delay > 0 && callback != nothing
        traces, weights = get_traces(pf_state), get_log_norm_weights(pf_state)
        for t=1:delay callback(t, obs_traj[t], traces, weights) end
    end
    # Compute times for each batch
    timesteps = collect(batch_size+delay:batch_size:n_obs)
    if timesteps[end] < n_obs push!(timesteps, n_obs) end
    # Feed new observations batch-wise
    for (batch_i, t) in enumerate(timesteps)
        if resample && get_ess(pf_state) < (n_particles * ess_threshold)
            @debug "Resampling..."
            pf_residual_resample!(pf_state, priority_fn=priority_fn)
            if rejuvenate != nothing rejuvenate(pf_state) end
        end
        t_prev = batch_i == 1 ? 0 : t - batch_size
        if update_proposal != nothing && (t - t_prev) > 1
            # Data-driven update if a sequence states are observed
            pf_update!(pf_state, (t, world_args...), argdiffs,
                       obs_choices[batch_i], update_proposal,
                       (t_prev+1, t, obs_traj[t_prev+1:t]))
        else
            # Standard update otherwise
            pf_update!(pf_state, (t, world_args...), argdiffs,
                       obs_choices[batch_i])
        end
        if callback != nothing # Run callback on current traces
            trs, ws = get_traces(pf_state), get_log_norm_weights(pf_state)
            callback(t, obs_traj[t], trs, ws)
        end
    end
    # Return particles and their weights
    traces, weights = get_traces(pf_state), get_log_norm_weights(pf_state)
    lml_est = logsumexp(get_log_weights(pf_state)) - log(n_particles)
    return traces, weights, lml_est
end


"Run goal inference via Sequential Inverse Plan Serach (SIPS) on a trajectory."
function run_sips_inference2(goal_idx, traj, goals, obs_terms,
                            world_init::WorldInit, world_config::WorldConfig)
    # Construct new dataframe for this trajectory
    df = DataFrame()

    # Set up logger and buffer to store logged messages
    log_buffer = IOBuffer() # Buffer of any logged messages
    logger = SimpleLogger(log_buffer)

    # Set up callback to collect data
    all_goal_probs = [] # Buffer of all goal probabilities over time
    true_goal_probs = Float64[] # Buffer of true goal probabilities
    step_times = Float64[] # Buffer of wall clock durations per step
    log_messages = String[] # Buffer of log messages for each timestep
    function data_callback(t, state, trs, ws)
        push!(step_times, time())
        println("Timestep $t")
        goal_probs = sort!(get_goal_probs(trs, ws, collect(1:length(goals))))
        push!(all_goal_probs, Vector{Float64}(collect(values(goal_probs))))
        push!(true_goal_probs, goal_probs[goal_idx+1])
        push!(log_messages, String(take!(log_buffer)))
    end

    # Set up rejuvenation moves
    rejuv_fns = Dict(
        :goal => pf -> pf_goal_move_accept!(pf, goals),
        :replan => pf -> pf_replan_move_accept!(pf),
        :mixed => pf -> pf_mixed_move_accept!(pf, goals; mix_prob=0.50),
        nothing => nothing
    )

    # Run a particle filter to perform online goal inference
    n_goals = length(goals)
    n_samples = SAMPLE_MULT * n_goals
    goal_strata = Dict((:goal_init => :goal) => collect(1:n_goals))
    start_time = time()
    with_logger(logger) do
        traces, weights = world_particle_filter2(
            world_init, world_config, traj, obs_terms, n_samples;
            resample=RESAMPLE, rejuvenate=rejuv_fns[REJUVENATE],
            callback=data_callback, strata=goal_strata)
    end

    # Process collected data
    all_goal_probs = reduce(hcat, all_goal_probs)'
    step_times = step_times .- start_time
    step_durs = step_times - [0; step_times[1:end-1]]
    states_visited = map(log_messages) do msg
        lines = split(msg,  r"\n|\r\n")
        count = 0
        for l in lines
            m = match(r".*Node Count: (\d+).*", l)
            if m == nothing continue end
            count += parse(Int, m.captures[1])
        end
        return count
    end

    # Add data to dataframe
    df.step_times = step_times
    df.step_durs = step_durs
    df.states_visited = states_visited
    df.true_goal_probs = true_goal_probs
    for (i, goal_probs) in enumerate(eachcol(all_goal_probs))
        df[!, "goal_probs_$(i-1)"] = goal_probs
    end

    # Return dataframe
    return df
end


"Run all experiments for a domain."
function run_domain_experiments2(path, domain_name, obs_subdir="optimal",
                                method=:sips)
    # Extract problem indices
    problem_fns = filter(fn -> occursin(r"problem_(\d+).pddl", fn),
                         readdir(joinpath(path, "problems", domain_name)))
    problem_idxs = [parse(Int, match(r".*problem_(\d+).pddl", fn).captures[1])
                    for fn in problem_fns]

    # Run experiments for each problem
    domain_dfs = []
    summary_df = DataFrame()
    for idx in problem_idxs
        println("Running experiments for problem $idx...")
        dfs, s_df = run_problem_experiments(path, domain_name, idx,
                                            obs_subdir, method)
        append!(summary_df, s_df)
        push!(domain_dfs, dfs)
    end

    # Compute and save summary statistics
    summary_stats = analyze_domain_results(summary_df)
    df_fn = "$(domain_name)_summary.csv"
    df_path = joinpath(path, "results", domain_name, df_fn)
    println("Writing domain summary results to $df_fn...")
    CSV.write(df_path, summary_stats)

    return summary_df, summary_stats
   
end

function foo(x)
    println("here is my x value:")
    println(x)
end