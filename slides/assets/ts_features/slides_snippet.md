---
background-image: url(assets/ts_features/frame_01.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Start from the hourly load. Raw material: the national load, one row per hour (`resample`). Point at the daily double peak and at the weekend dip: that is what the lags will exploit.

---
background-image: url(assets/ts_features/frame_02.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Pick a prediction time and a horizon. One row of the training table = one prediction time p. The target is `load_mw.shift(-h)`. We repeat the construction for h = 1, 12 and 24 h.

---
background-image: url(assets/ts_features/frame_03.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Calendar features of the target time. 5 features computed on the *target* time, in Europe/Paris time: hour of day, day of week, day of year, year and public holiday flag.

---
background-image: url(assets/ts_features/frame_04.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Recent load lags. For h = 12 the most recent usable observation is p itself (= lag_12). Lags 1 to 11 would peek into the future, so they are not built. This is why the features are built per horizon.

---
background-image: url(assets/ts_features/frame_05.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Same clock hour, earlier days. Exploits the daily and weekly seasonality. `load_same_hour` adds lags 36 and 180: the same clock hour as p, one day and one week before. 17 load lags in total for h = 12.

---
background-image: url(assets/ts_features/frame_06.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Hourly changes of the load. `load_diffs`: lag_k - lag_(k+1) for k = h ... 22, so 11 features for h = 12. A tree cannot subtract two columns itself, so the differences are given explicitly.

---
background-image: url(assets/ts_features/frame_07.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Load now versus earlier loads. `load_diffs` adds lag_h - lag_24, lag_48, lag_168 and `load_same_hour` adds lag_h - lag_36, lag_180: 5 more features, 16 load differences in total.

---
background-image: url(assets/ts_features/frame_08.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Rolling median and IQR. median and IQR over 24 h and 168 h windows ending at p and at t - 24 h: 2 stats x 2 widths x 2 end points = 8 features.

---
background-image: url(assets/ts_features/frame_09.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Weather forecasts. Weather *forecasts* rather than observations: a forecast issued at least 24 h before the valid time is assumed available at p (day-1 forecast for h <= 18 h, day-2 above). One feature per city, 3 of 10 shown.

---
background-image: url(assets/ts_features/frame_10.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Rolling temperature means. 9 features per city (temperature at t + 8 rolling means) x 10 cities = 90 weather features: the biggest group by far.

---
background-image: url(assets/ts_features/frame_11.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- National temperature changes. `temp_diffs`: change of the national mean temperature over 24 h, 168 h and h, and anomalies against its 24 h and 168 h means. Windows end at t and only forecasts valid by t are used. 5 features that combine the cities, which a tree cannot do itself.

---
background-image: url(assets/ts_features/frame_12.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- One feature row per prediction time. Column names are abbreviated. `prediction_time` and `target_time` are dropped and everything is converted to floats: the tabular models see no timestamp, the time structure only lives in the features. These are the round-2 feature groups.

---
background-image: url(assets/ts_features/frame_13.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Slide p forward: a plain tabular dataset. Number of features per horizon: 163 (h = 1), 141 (h = 12), 110 (h = 24). Without the three feature groups: 129, 118, 102. Evaluation uses walk-forward CV with a gap between the train and test periods.

---
background-image: url(assets/ts_features/frame_14.png)
background-position: center
background-repeat: no-repeat
background-size: 100% 100%

???

- Same table, three quantile models. HGB: one model per quantile. TabICL and Kumo-Tabular: `fit` just stores (X, y); prediction runs the pretrained transformer with the training table as context. Kumo returns 999 quantiles, we keep 5/50/95%.
