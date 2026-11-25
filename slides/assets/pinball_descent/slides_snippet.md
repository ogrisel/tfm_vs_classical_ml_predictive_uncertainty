---
background-image: url(assets/pinball_descent/frame_01.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- A skewed sample, 5 quantiles to estimate. Synthetic data: y = exp(0.75 z) with z standard normal, so the median is about 1 and the mean about 1.3. We want 5 numbers q such that a fraction tau = 5, 25, 50, 75 and 95% of the sample lies below q. Sorting the sample would do for a fixed sample, but a quantile regression model has to learn q(x) with gradients: we need a loss whose minimizer is the quantile.

---
background-image: url(assets/pinball_descent/frame_02.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- The pinball loss of one observation. A single observation y = 1. Predicting too low costs tau per unit of error, predicting too high costs 1 - tau. For tau = 5% overshooting is 19 times more expensive than undershooting, which pushes q down; for 95% it is the opposite. At tau = 50% both sides cost the same: this is half the absolute error.

---
background-image: url(assets/pinball_descent/frame_03.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Averaged over the sample: convex, minimum at the quantile. Averaging over the 200 observations gives one convex curve per tau. Its slope at q is (share of the sample below q) - tau: it vanishes exactly when the share below q equals tau, i.e. at the sample tau-quantile. So the quantile is the minimizer. We could scan the curve; instead we only use the slope at the current q, which is what gradient descent (and the training of a model) has access to.

---
background-image: url(assets/pinball_descent/frame_04.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Why the curve looks smooth: zoom on the slope. Zoom on the median (50%) curve, on a window that contains 12 observations. Top row: the mean loss, bottom row: its slope, which is the share of the sample below q minus tau, i.e. the empirical CDF shifted by tau. Each observation makes the slope jump by 1/n = 0.005, so the loss has a kink there, but a kink that small is invisible at full scale: the staircase looks like a smooth curve and its integral, the loss, looks smooth too. Also note that the slope crosses zero exactly where the share below q reaches tau.

---
background-image: url(assets/pinball_descent/frame_05.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Start from the prediction q = 0. Initialization: q = 0 for the 5 levels, as for a model that outputs zeros. Every observation is above q, so the share below is 0 and the slope is 0 - tau = -tau: the steeper for the high quantiles. The tangent drawn on each curve shows that increasing q reduces the loss.

---
background-image: url(assets/pinball_descent/frame_06.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Iteration 1: step by tau. First update: nothing is below q, so every quantile moves up by eta * tau: 0.05, 0.25, 0.50, 0.75, 0.95. High quantiles take the biggest steps because they are the furthest from 0.

---
background-image: url(assets/pinball_descent/frame_07.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Iteration 2: the steps start to differ. Some observations are now below the 25 to 95% predictions, so the gap between the share below and tau, hence the step, shrinks. The 5% quantile still has nothing below it: it takes the same tiny step of 0.05 again. The steps are never larger than tau: the update is bounded whatever the scale of y.

---
background-image: url(assets/pinball_descent/frame_08.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Iteration 3: the middle quantiles settle. The 25 and 50% quantiles are within about 10% of their final value and the tangent on their curves is getting flat; the 75 and 95% quantiles are still travelling. The updates are driven by the gap between the share below q and tau: no sorting, no assumption on the distribution.

---
background-image: url(assets/pinball_descent/frame_09.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Iteration 10: the tails are still moving. After 10 iterations the 25, 50 and 75% quantiles are done. The 5 and 95% tails keep moving: few observations sit around them, so q has to travel across a low density region and the share below changes slowly. Better optimizers (momentum, Adam, line search) or a step size matched to the data scale speed this up.

---
background-image: url(assets/pinball_descent/frame_10.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Iteration 50: five quantiles, a 90% interval. After 50 iterations, the predictions match the minimizers of the mean pinball loss (the sample quantiles, hollow markers), the upper tail being the slowest to converge. The 5% and 95% quantiles delimit a 90% prediction interval. A quantile regression model does the same, with q = f(x; theta) and the gradient flowing into theta.
