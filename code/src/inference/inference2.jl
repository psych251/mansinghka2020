export world_particle_filter3

using GenParticleFilters

include("utils.jl")
include("updates.jl")
include("kernels.jl")

"Online inference over a world model using a particle filter."
function world_particle_filter3(
        world_init::WorldInit, world_config::WorldConfig,
        obs_traj::Vector{State}, obs_terms::Vector{<:Term}, n_particles::Int;
        batch_size::Int=1, delay::Int=0, strata=nothing, callback=nothing,
        ess_threshold::Float64=1/4, update_proposal=nothing,
        priority_fn=w->w*0.75, resample=true, rejuvenate=nothing)
    println("Running Particle Filter 3 -- Reimplementation")
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