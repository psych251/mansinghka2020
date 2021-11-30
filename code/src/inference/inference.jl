export world_importance_sampler, world_particle_filter, world_particle_filter2

using GenParticleFilters

include("utils.jl")
include("updates.jl")
include("kernels.jl")

"Generate weighted importance samples over a world model."
function world_importance_sampler(
        world_init::WorldInit, world_config::WorldConfig,
        obs_traj::Vector{State}, obs_terms::Vector{<:Term}, n_samples::Int;
        use_proposal=true, strata=nothing, callback=nothing)
    @unpack domain, planner = world_config
    # Construct choicemaps from observed trajectory
    n_obs = length(obs_traj)
    obs_choices = traj_choicemaps(obs_traj, domain, obs_terms;
                                  as_choicemap=true)
    # Initialize traces of world model
    init_traces, _ = strata == nothing ?
        unzip(generate(init_world_model, (world_init,)) for i in 1:n_samples) :
        enumerate_traces(init_world_model, (world_init,), strata)
    # Compute importance sample for each initial trace
    traces = Vector{Trace}(undef, n_samples)
    weights = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        init_idx = mod(i-1, length(init_traces)) + 1
        world_state = get_retval(init_traces[init_idx])
        init_choices = get_choices(init_traces[init_idx])
        @unpack env_state, plan_state, goal_state = world_state
        # Set-up data-driven proposal associated with the planner
        prop_args = (1, n_obs, plan_state, planner, domain,
                     env_state, goal_state, obs_traj, fill(nothing, n_obs))
        # Propose choices
        prop_choices, prop_weight, _ = use_proposal ?
            propose(propose_step_range, prop_args) : (choicemap(), 0.0, 0.0)
        constraints = merge(obs_choices, init_choices, prop_choices)
        # Sample from model given constraints
        model_args = (n_obs, world_init, world_config)
        traces[i], model_weight = generate(world_model, model_args, constraints)
        weights[i] = model_weight - prop_weight
    end
    # Normalize weights
    lml_est = logsumexp(weights) - log(n_samples)
    weights = lognorm(weights)
    # Run callback on sampled traces
    if callback != nothing
        callback(n_obs, obs_traj[end], traces, weights)
    end
    # Return traces and their weights
    return traces, weights, lml_est
end

"Online inference over a world model using a particle filter."
function world_particle_filter(
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

"Online inference over a world model using a particle filter."
function particle_filter_reimplementation(
    world_init::WorldInit,
    world_config::WorldConfig,
    observed_trajectory::Vector{State},
    observed_terms::Vector{<:Term},
    num_particles::Int;
    strata=nothing,
    callback=nothing,
    resample=true,
    rejuvenate=nothing)
    
    threshold::Float64=1/4
    @unpack domain = world_config
    num_observations = length(observed_trajectory) 
    time_steps = collect(1:num_observations)
    observed_traj_choices = traj_choicemaps(observed_trajectory, 
                                            domain, 
                                            observed_terms)
    
    world_args = (world_init, world_config)
    argdiffs = (UnknownChange(), NoChange(), NoChange()) # Gen types, 
                                                         # UknownChannge: no information of change is provided
                                                         # NoChange: value is unchanged
    
    # if strata=nothing then this simply calls initialize_particle_filter
    beleif_state =  initialize_pf_stratified(world_model,    # world model
                                         (0, world_args...), # the initial arugments for the world model
                                         choicemap(),        # initial observations expressed as choicemap
                                         strata,             # goal_strata
                                         num_particles)      # number of particles
    
    # iterate through each timestep
    for (time_step, t) in enumerate(time_steps)
        t_prev = t-1
        if time_step == 1
            t_prev = 0
        end
        
        # get_ess computes effective samples size of particles in filter
        # so if effective sample size if less than some threshold vlaue of num_particles than resample and rejuvenate 
        # (see psudocode code)
        if resample && get_ess(beleif_state) < (num_particles * threshold) 
            @debug "Resampling..."
            pf_residual_resample!(beleif_state, priority_fn=w->w*0.75)
            if rejuvenate != nothing 
                rejuvenate(beleif_state) 
            end
        end
        
        # update beleif state
        pf_update!(beleif_state, 
                   (t, world_args...), 
                   argdiffs, 
                   observed_traj_choices[time_step])
         
        # compute traces and weights and normalize (see psudocode)
        if callback != nothing 
            traces_t = get_traces(beleif_state)
            weights_t = get_log_norm_weights(beleif_state)
            callback(t, observed_trajectory[t], traces_t, weights_t)
        end
    end
        
    # compute traces and weights and normalize (see psudocode)
    traces = get_traces(beleif_state)
    weights = get_log_norm_weights(beleif_state)
    lml_est = logsumexp(get_log_weights(beleif_state)) - log(num_particles)    
    return traces, weights, lml_est
end

