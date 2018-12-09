import progressbar as pb

from hyperparameters import *


# TODO: Rename states

def train(maddpg, n_episodes=1000, max_t=1000, save_every=50):
    widget = [
        "Episode: ", pb.Counter(), '/' , str(n_episodes), ' ', 
        pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', 
        'Rolling Average: ', pb.FormatLabel('')
    ]
    timer = pb.ProgressBar(widgets=widget, maxval=n_episodes).start()

    solved = False
    scores_total = []
    scores_deque = deque(maxlen=100)
    rolling_score_averages = []
    best_score = 0.0
    
    for i_episode in range(1, n_episodes+1):
        current_average = 0.0 if i_episode == 1 else rolling_score_averages[-1]
        widget[12] = pb.FormatLabel(str(current_average)[:6])
        timer.update(i_episode)
            
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations[:, -STATE_SIZE:]
        scores = np.zeros(num_agents)
        maddpg.reset()
        
        # for t in range(max_t):
        while True:
            actions = maddpg.act(states)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations[:, -STATE_SIZE:]
            rewards = env_info.rewards
            dones = env_info.local_done
            
            maddpg.step(states, actions, rewards, next_states, dones)
            
            scores += rewards
            states = next_states
            
            if np.any(dones):
                break
        
        max_episode_score = np.max(scores)
        
        scores_deque.append(max_episode_score)
        scores_total.append(max_episode_score)

        average_score = np.mean(scores_deque)
        rolling_score_averages.append(average_score)
        
        if average_score > best_score:
            best_score = average_score
            if solved:
                # This model is better than a previously saved model, so we'll save it
                maddpg.save_model()
        
        if average_score >= 0.5 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, average_score
            ))
            solved = True
            maddpg.save_model()
        
        if i_episode % save_every == 0 and not solved:
            maddpg.save_model()

    maddpg.save_model()

    return scores_total, rolling_score_averages
