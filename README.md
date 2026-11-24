# TabICL vs scikit-learn for predictive uncertainty quantification

Slides for a talk at [Compute! Paris 2026](https://compute.events/paris2026/).

- **Speaker:** Olivier Grisel (Probabl)
- **Co-author of the case study:** Jérôme Dockès (Probabl)
- **When:** Wednesday 2026-11-25, 10:50–11:20 (30 min), Auditorium
- **Track:** Models on Trial (Talk), tags: MachineLearning, ScientificComputing
- **Talk page:** <https://pretalx.com/compute-paris-2026/talk/MKRCE8/>
  (also reachable from the
  [schedule](https://compute.events/paris2026/schedule.html#talk/MKRCE8))
- **Venue:** Hôtel du Port, Nogent-sur-Marne (moved from the CICSU, Jussieu
  campus, due to a student blockade; see the schedule page).

## Abstract

TabICL is a fully-open source, state of the art Tabular Foundational Model
(TFM). TFMs are pretrained on large amount of synthetic data and have been
advertised as being able to provide well calibrated probabilistic predictions
both for classification and regression tasks.

In this presentation, we will present the results of a comparative study of the
predictive capabilities of TabICL versus several scikit-learn based
alternatives. In particular, we will contrast the following aspects: simplicity
of development, computational and statistical performance in small and large
sample size regimes. Statistical performance will be evaluated with strictly
proper scoring rules that assess both the calibration and sharpness of
predictions on left out data.

### Description

Our case study will focus on a time series forecasting problem, which can be
considered as a grid of probabilitic regression problems at different horizons.

We will present how tabular models such as gradient boosted trees and open
source & weights TFMs such as TabICL and Kumo can be combined with
time-structured feature engineering to solve this task and how the compare
against Time Series Foundational Models.

## Source material

The narrative is illustrated with the electricity load forecasting case study
by Jérôme Dockès:

- Published reports: <https://probabl-ai.github.io/skore-example-electricity-load/>
- Code: <https://github.com/probabl-ai/skore-example-electricity-load>

## Slides

The deck is written with [remark.js](https://remarkjs.com) and styled after
[probabl.ai](https://www.probabl.ai) (beige / deep-blue / orange palette, serif
headings).

```
slides/
├── index.html        # remark config + slide source (Markdown in a <textarea>)
├── style.css         # Probabl-inspired theme
├── assets/           # Probabl logo (dark and white variants)
└── vendor/remark.min.js   # vendored so the talk works offline
```

To view or present, just open `slides/index.html` in a browser (no server
needed). Useful keys: `P` presenter mode (notes + timer), `C` clone the view to
a second window, `F` fullscreen, `?` help.

## Status

Draft outline only: each slide has a title, a few bullets and speaker notes
that say where the content comes from. Items marked `TODO` need your decision
or additional material.
