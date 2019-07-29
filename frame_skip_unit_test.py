frame_skip = 4

num_steps = 100_000

take_actions = []
for i in range(num_steps):
    if i % frame_skip == 0:
        take_actions.append(i)

print('num_steps / frame_skip =', num_steps / frame_skip)
print('len(take_actions):', len(take_actions))
