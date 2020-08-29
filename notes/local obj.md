Side experiment with local objective and global x

Setting:
- agents on a 2d grid
- initialize all agents with one random x
- objective is a quadratic function (x²+y²) with a randomly picked minimum for each agent.
- communication protocoll is again distance based, but the maximum transmission probability is severely clipped (i think somewhere at 1% succes probability)

Results:
- extremely fast convergence to the mean (couple of seconds)

Thoughts:
- communication is basically irrelevant, as we just move to the mean and if the positions are outdated this will not cause any problems. Also, unlike the other setting, the whole x moves to one very clearly specified x' which is independent of the movement of the other robots.
- the goal is basically to find the mean of points.
- in the specific objective, the gradient also points directly to the minimum as the the quadratic function is strictly convex. When choosing other objectives the convergence might be much slower and more interesting to look at.

Outlook:
- other objective functions
- draw the global objective \sum of f_i(x) in the plot