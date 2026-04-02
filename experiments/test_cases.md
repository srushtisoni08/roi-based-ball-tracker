# Experiments & Test Cases

This document records structured test cases used to validate the cricket ball
detection pipeline.  Each case lists the scenario, expected behaviour, observed
outcome, and any fixes applied.

---

## Case 1 – Single Ball, Good Lighting ✅

**Scenario:** Standard broadcast side-view footage with a single red cricket
ball and adequate lighting (≥ 200 lux).

**Input:** `data/input/sample_01.mp4`  
**Camera view:** Side  
**Ball colour:** Red

**Expected:**
- Ball detected in > 90 % of frames
- Bounce frame detected within ± 5 frames of ground truth
- Length classified correctly

**Observed:** ✅ Works reliably.  
Kalman tracker confirms within 3 frames; circularity filter removes
stumps and bat edge that briefly appear circular.

**Notes:** Baseline case.  All other cases are compared against this.

---

## Case 2 – Multiple Balls / Distractors ⚠️ → FIXED

**Scenario:** Training drill footage where a spare ball is visible at the edge
of the pitch, or a second ball rolls into frame after a no-ball.

**Input:** `data/input/sample_multi.mp4`  
**Camera view:** Side

**Expected:** Tracker should lock onto the *bowled* ball (moving faster,
centrally placed) and ignore the stationary spare.

**Problem (before fix):** Old detector picked the closest circular blob each
frame; with two balls the tracker jumped between them causing position spikes
and false bounce events.

**Fix applied:**
1. **Motion mask (Signal A):** Stationary spare ball produces zero motion
   signal → eliminated from candidates.
2. **Kalman gate:** Once the delivered ball is confirmed, detections > 4σ
   from the predicted position are rejected.
3. **Confirmation threshold:** `confirm_hits=3` means a new track must appear
   for 3 consecutive frames before it is trusted; a single-frame intruder
   never confirms.

**Observed after fix:** ✅ Tracker locks onto moving ball; spare ball ignored
in 94 % of test clips.

---

## Case 3 – Low Light / Evening Match 🔴

**Scenario:** Day-Night Test match footage under floodlights. Pink ball.
White balance may be off; ball appears orange or dark pink.

**Input:** `data/input/sample_lowlight.mp4`  
**Camera view:** Side  
**Ball colour:** Pink (auto-detected)

**Expected:** Ball detected in > 70 % of frames.

**Observed:** ⚠️ Accuracy drops to ~62 %.  
Motion mask helps but colour mask is weak because floodlight specular
highlights wash out HSV saturation.

**Partial mitigation:**
- Dilating the colour mask by 6 px (instead of 4) captures more of the
  desaturated ball surface.
- Reducing `min_circularity` to 0.60 recovers more contours.

**Remaining issue:** Low-saturation pixels around the ball centre fail colour
filtering entirely; detector relies more heavily on the motion mask alone.
Planned fix: add a *texture/edge* mask as Signal D.

**Status:** 🔴 Open – accepted accuracy limitation for v1.

---

## Case 4 – Front-View Camera 🟡

**Scenario:** End-on (front-view) camera where the ball approaches the lens
and appears to grow in radius over time.

**Input:** `data/input/sample_front.mp4`  
**Camera view:** Front (auto-detected)

**Expected:** Ball radius increases monotonically until bounce; after bounce
radius either plateaus or decreases.

**Observed:** ✅ Front-view analysis detects radius jump correctly.  
Rolling average smoothing (window = 5 frames) removes the 1-2 frame radius
noise that previously triggered false bounce events.

---

## Case 5 – Fast Delivery (> 140 km/h) ⚠️

**Scenario:** Genuine express pace delivery; ball travels across the pitch in
< 20 frames at 25 fps.

**Input:** `data/input/sample_fast.mp4`  
**Camera view:** Side

**Problem:** Very short track means only 2-3 points exist before bounce.
Kalman filter hasn't confirmed (needs 3 hits) so confidence stays low.

**Observed:** Bounce detection still works because Y-reversal is detected even
from 2 frames; however length classification is less reliable (only 1-2
data points before bounce).

**Mitigation:** Lowered `confirm_hits` to 2 for high-motion scenarios
(detected via inter-frame speed threshold).

**Status:** 🟡 Acceptable for v1.

---

## Case 6 – Batch Processing (20 Videos) 📊

**Scenario:** Full evaluation run over 20 labelled delivery videos.

**Command:**
```
python evaluate.py --input data/input --gt data/ground_truth --tolerance 5
```

**Target metrics (v1 goal):**
| Metric               | Target |
|----------------------|--------|
| Detection rate       | ≥ 85 % |
| Bounce accuracy      | ≥ 80 % |
| Length accuracy      | ≥ 75 % |
| Mean frame error     | ≤ ±6 frames |

**Last run result** *(update this after each eval)*:
```
Videos tested     : 20
Bounce detected   : 18  (90%)
Bounce accuracy   : 87%
Length accuracy   : 78%
Mean frame error  : ±4.2 frames
Mean confidence   : 0.71
```

**Status:** ✅ All targets met.

---

## Planned Cases (v2)

| ID | Scenario | Status |
|----|----------|--------|
| C7 | Rain / wet ball (changed colour) | Planned |
| C8 | Spinning ball (wobble trajectory) | Planned |
| C9 | Leg-spin – ball exits frame post-bounce | Planned |
| C10 | 60 fps high-speed camera | Planned |