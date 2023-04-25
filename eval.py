import time



def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.
			Parameters:
				None
			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		
    
def rollout(policy, env, render):
	
    s = env.reset()
    done = False

    t = 0

    ep_len = 0
    ep_ret = 0

    for i in range(1000):
        t += 1

        env.render()
        # time.sleep(0.1)
        

        action = policy(s).detach().numpy()
        s, r, _, done,_ = env.step(action)
        

        ep_ret += r
        
        if done:
            break

    ep_len = t

    env.close()

    # yield ep_len, ep_ret
    return ep_len, ep_ret

    # while True:
    #     s, _ = env.reset()
    #     done = False

    #     t = 0

    #     ep_len = 0
    #     ep_ret = 0

    #     while not done:
    #         t += 1

    #         if render:
    #             env.render()
    #         time.sleep(0.1)

    #         action = policy(s).detach().numpy()
    #         s, r, _, done,_ = env.step(action)

    #         ep_ret += r

    #     ep_len = t

    #     # yield ep_len, ep_ret
    #     return ep_len, ep_ret


def eval_policy(policy, env, render=True):
      
      for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
            _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
                