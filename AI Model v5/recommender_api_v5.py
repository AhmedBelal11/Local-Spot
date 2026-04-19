"""
Hybrid Recommender API — V5
LightGBM + ALS + Popularity Cold Start
========================================
Setup:
    pip install flask lightgbm pandas numpy

Files needed (from model_files_v5.zip):
    lightgbm_model_v5.txt       features_v5.json
    product_lookup_v5.pkl       user_profiles_v5.pkl
    user_gender_map_v5.pkl      als_user_factors_v5.pkl
    als_product_factors_v5.pkl  als_user_idx_v5.pkl
    als_product_idx_v5.pkl      popular_men_v5.pkl
    popular_women_v5.pkl        lgb_subcat_encoder_v5.pkl
    lgb_gender_encoder_v5.pkl   lgb_season_encoder_v5.pkl

Run:
    python recommender_api_v5.py

Endpoints:
    GET  /health
    POST /interact
    POST /recommend
    GET  /user/<user_id>
"""

import os, math, pickle, json, random
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

# ── Config ────────────────────────────────────────────────────────
MODEL_DIR      = os.getenv("MODEL_DIR", ".")
DECAY_HALFLIFE = 60
DECAY_LAMBDA   = math.log(2) / DECAY_HALFLIFE
LGB_WEIGHT     = 0.60
ALS_WEIGHT     = 0.40
SUBCAT_CAP     = 2
CANDIDATE_POOL = 50
DEFAULT_TOP_K  = 5

EVENT_WEIGHTS = {
    "purchase":  3.0,
    "addtocart": 2.0,
    "view_long": 1.0,
    "view":      0.5,
}

# ── Load Model Files ──────────────────────────────────────────────
def _load(fname):
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        return pickle.load(f)

print("Loading model files...")

booster              = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model_v5.txt"))
feature_cols         = json.load(open(os.path.join(MODEL_DIR, "features_v5.json")))
product_lookup       = _load("product_lookup_v5.pkl")
user_profiles_df     = _load("user_profiles_v5.pkl")
user_gender_map      = _load("user_gender_map_v5.pkl")
als_user_factors     = _load("als_user_factors_v5.pkl")
als_product_factors  = _load("als_product_factors_v5.pkl")
als_user_idx         = _load("als_user_idx_v5.pkl")
als_product_idx      = _load("als_product_idx_v5.pkl")
popular_men          = _load("popular_men_v5.pkl")
popular_women        = _load("popular_women_v5.pkl")
subcat_enc           = _load("lgb_subcat_encoder_v5.pkl")
gender_enc           = _load("lgb_gender_encoder_v5.pkl")
season_enc           = _load("lgb_season_encoder_v5.pkl")

# Fast lookups
known_products    = set(product_lookup.keys())
all_subcats       = list({v["subcategory"] for v in product_lookup.values()})
user_profiles_idx = (
    user_profiles_df.set_index("user_id").to_dict(orient="index")
    if len(user_profiles_df) > 0 else {}
)

_prices   = [v["price"] for v in product_lookup.values()]
PRICE_MEAN = float(np.mean(_prices))

print(f"  Products      : {len(product_lookup)}")
print(f"  Known users   : {len(user_profiles_idx)}")
print("  Ready ✓")

# ── In-Memory Interaction Store ───────────────────────────────────
# { user_id → [ {product_id, event_type, timestamp, event_w, decay_w, combined_w} ] }
interaction_store = defaultdict(list)


def _decay(ts):
    now    = datetime.now(timezone.utc)
    ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    days   = max((now - ts_utc).total_seconds() / 86400, 0)
    return math.exp(-DECAY_LAMBDA * days)


def _refresh_decay(uid):
    for e in interaction_store[uid]:
        e["decay_w"]    = _decay(e["timestamp"])
        e["combined_w"] = e["decay_w"] * e["event_w"]


# ── Gender Filter ─────────────────────────────────────────────────
def _gender_ok(product_gender, user_gender):
    if product_gender == "Unisex" or user_gender is None:
        return True
    return product_gender == user_gender


# ── User Profile Builder ──────────────────────────────────────────
def _build_profile(uid):
    base = user_profiles_idx.get(uid)
    _refresh_decay(uid)
    live = interaction_store.get(uid, [])

    # Cold start
    if not live and base is None:
        return {
            "fav_subcategory": 0,
            "fav_gender":      0,
            "fav_season":      0,
            "fav_price_tier":  1,
            "avg_price":       PRICE_MEAN,
            "purchase_count":  1,
            "purchased_ids":   set(),
            "_source":         "cold-start",
        }

    # Accumulate live events
    subcat_w = defaultdict(float)
    gender_w = defaultdict(float)
    season_w = defaultdict(float)
    ptier_w  = defaultdict(float)
    price_sum = weight_sum = purchase_cnt = 0.0
    purchased = set()

    for e in live:
        pid = e["product_id"]
        if pid not in product_lookup:
            continue
        p = product_lookup[pid]
        w = e["combined_w"]
        subcat_w[p["subcategory_encoded"]] += w
        gender_w[p["gender_encoded"]]      += w
        season_w[p["season_encoded"]]      += w
        ptier_w [p["price_tier"]]          += w
        price_sum  += p["price"] * w
        weight_sum += w
        if e["event_type"] == "purchase":
            purchase_cnt += 1
            purchased.add(pid)

    if weight_sum > 0:
        fs  = max(subcat_w, key=subcat_w.get)
        fg  = max(gender_w, key=gender_w.get)
        fss = max(season_w, key=season_w.get)
        ft  = max(ptier_w,  key=ptier_w.get)
        ap  = price_sum / weight_sum
        if base:
            purchase_cnt += int(base.get("purchase_count", 0))
    else:
        fs  = int(base["fav_subcategory"])
        fg  = int(base["fav_gender"])
        fss = int(base["fav_season"])
        ft  = int(base["fav_price_tier"])
        ap  = float(base["avg_price"])
        purchase_cnt = int(base["purchase_count"])

    return {
        "fav_subcategory": fs,
        "fav_gender":      fg,
        "fav_season":      fss,
        "fav_price_tier":  ft,
        "avg_price":       ap,
        "purchase_count":  max(int(purchase_cnt), 1),
        "purchased_ids":   purchased,
        "_source":         "live" if weight_sum > 0 else "training",
    }


# ── Candidate Retrieval ───────────────────────────────────────────
def _get_candidates(history, user_gender, exclude_ids, topk=CANDIDATE_POOL):
    sub_list = [product_lookup[p]["subcategory"] for p in history if p in product_lookup]

    ok = lambda pid: pid not in exclude_ids and _gender_ok(product_lookup[pid]["gender"], user_gender)

    if not sub_list:
        pool = [p for p in known_products if ok(p)]
        random.shuffle(pool)
        return pool[:topk]

    top_subs   = [s for s, _ in Counter(sub_list).most_common(3)]
    other_subs = [s for s in all_subcats if s not in top_subs]

    primary   = [p for p, d in product_lookup.items() if d["subcategory"] in top_subs   and ok(p)]
    secondary = [p for p, d in product_lookup.items() if d["subcategory"] in other_subs  and ok(p)]
    random.shuffle(primary)
    random.shuffle(secondary)

    n_primary  = int(topk * 0.70)
    candidates = primary[:n_primary] + secondary[:topk - n_primary]

    if len(candidates) < topk:
        extra = [p for p in known_products if p not in set(candidates) | exclude_ids and ok(p)]
        random.shuffle(extra)
        candidates += extra[:topk - len(candidates)]

    return candidates[:topk]


# ── Hybrid Ranking ────────────────────────────────────────────────
def _rank_hybrid(candidates, profile, uid, topk=DEFAULT_TOP_K):
    fs  = int(profile["fav_subcategory"])
    fg  = int(profile["fav_gender"])
    fss = int(profile["fav_season"])
    ft  = int(profile["fav_price_tier"])
    ap  = float(profile["avg_price"])
    pc  = int(profile["purchase_count"])

    feat_rows  = []
    valid_pids = []
    for pid in candidates:
        if pid not in product_lookup:
            continue
        p = product_lookup[pid]
        feat_rows.append({
            "subcategory_encoded": p["subcategory_encoded"],
            "gender_encoded":      p["gender_encoded"],
            "season_encoded":      p["season_encoded"],
            "price":               p["price"],
            "price_normalized":    p["price_normalized"],
            "price_tier":          p["price_tier"],
            "fav_subcategory":     fs,
            "fav_gender":          fg,
            "fav_season":          fss,
            "fav_price_tier":      ft,
            "avg_price":           ap,
            "purchase_count":      pc,
            "price_diff":          abs(p["price"] - ap) / (ap + 1),
            "subcategory_match":   int(p["subcategory_encoded"] == fs),
            "gender_match":        int(p["gender_encoded"]      == fg),
            "season_match":        int(p["season_encoded"]      == fss),
            "price_tier_match":    int(p["price_tier"]          == ft),
        })
        valid_pids.append(pid)

    if not feat_rows:
        return []

    df        = pd.DataFrame(feat_rows)
    lgb_scores = booster.predict(df[feature_cols].values)

    # ALS scores
    u_idx      = als_user_idx.get(uid)
    als_scores = np.zeros(len(valid_pids))
    if u_idx is not None:
        user_vec = als_user_factors[u_idx]
        for i, pid in enumerate(valid_pids):
            p_idx = als_product_idx.get(pid)
            if p_idx is not None:
                als_scores[i] = float(np.dot(user_vec, als_product_factors[p_idx]))
        a_min, a_max = als_scores.min(), als_scores.max()
        if a_max > a_min:
            als_scores = (als_scores - a_min) / (a_max - a_min)

    hybrid = LGB_WEIGHT * lgb_scores + ALS_WEIGHT * als_scores

    df["_pid"]  = valid_pids
    df["score"] = hybrid
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    results    = []
    sub_counts = {}
    for _, row in df.iterrows():
        pid = int(row["_pid"])
        sub = product_lookup[pid]["subcategory"]
        if sub_counts.get(sub, 0) < SUBCAT_CAP:
            p = product_lookup[pid]
            results.append({
                "product_id":  pid,
                "name":        p["name"],
                "category":    p["category"],
                "subcategory": sub,
                "gender":      p["gender"],
                "season":      p["season"],
                "price":       p["price"],
                "score":       round(float(row["score"]), 6),
            })
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
        if len(results) == topk:
            break

    return results


# ── Popularity Cold Start ─────────────────────────────────────────
def _popular_for_gender(gender, topk=DEFAULT_TOP_K):
    pool = popular_men if gender == "Men" else popular_women if gender == "Women" else popular_men
    results    = []
    sub_counts = {}
    for pid in pool:
        if pid not in product_lookup:
            continue
        p   = product_lookup[pid]
        sub = p["subcategory"]
        if sub_counts.get(sub, 0) < SUBCAT_CAP:
            results.append({
                "product_id":  pid,
                "name":        p["name"],
                "category":    p["category"],
                "subcategory": sub,
                "gender":      p["gender"],
                "season":      p["season"],
                "price":       p["price"],
                "score":       None,
            })
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
        if len(results) == topk:
            break
    return results


# ── Flask App ─────────────────────────────────────────────────────
app = Flask(__name__)


def _err(msg, status):
    return jsonify({"error": msg}), status


def _int_field(data, key, default=None, min_val=None, max_val=None):
    raw = data.get(key, default)
    if raw is None:
        return None, f"'{key}' is required"
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return None, f"'{key}' must be an integer"
    if min_val is not None and val < min_val:
        return None, f"'{key}' must be >= {min_val}"
    if max_val is not None and val > max_val:
        return None, f"'{key}' must be <= {max_val}"
    return val, None


# GET /health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "products":      len(product_lookup),
        "known_users":   len(user_profiles_idx),
        "event_weights": EVENT_WEIGHTS,
        "hybrid_weights": {"lightgbm": LGB_WEIGHT, "als": ALS_WEIGHT},
    }), 200


# POST /interact
@app.route("/interact", methods=["POST"])
def interact():
    """
    Log a user interaction.
    Body: { user_id, product_id, event_type, timestamp (optional) }
    event_type: view | view_long | addtocart | purchase
    """
    data = request.get_json(silent=True)
    if not data:
        return _err("Request body must be valid JSON", 400)

    uid, err = _int_field(data, "user_id")
    if err: return _err(err, 422)

    pid, err = _int_field(data, "product_id")
    if err: return _err(err, 422)
    if pid not in product_lookup:
        return _err(f"product_id {pid} not found", 404)

    event_type = data.get("event_type")
    if not event_type or event_type not in EVENT_WEIGHTS:
        return _err(f"event_type must be one of {list(EVENT_WEIGHTS.keys())}", 422)

    ts_raw = data.get("timestamp")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(str(ts_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return _err("timestamp must be ISO-8601, e.g. '2026-04-14T18:30:00'", 422)
    else:
        ts = datetime.now(timezone.utc)

    event_w = EVENT_WEIGHTS[event_type]
    decay_w = _decay(ts)
    entry   = {
        "product_id": pid,
        "event_type": event_type,
        "timestamp":  ts,
        "event_w":    event_w,
        "decay_w":    decay_w,
        "combined_w": event_w * decay_w,
    }
    interaction_store[uid].append(entry)

    return jsonify({
        "status":           "logged",
        "user_id":          uid,
        "product_id":       pid,
        "event_type":       event_type,
        "event_weight":     event_w,
        "decay_weight":     round(decay_w, 4),
        "combined_weight":  round(event_w * decay_w, 4),
    }), 200


# POST /recommend
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Get top-k recommendations for a user.
    Body: { user_id, top_k (optional, default 5) }

    Flow:
      1. Get registered gender
      2. If new user → popularity cold start
      3. If existing user → build profile → candidates → hybrid rank
    """
    data = request.get_json(silent=True)
    if not data:
        return _err("Request body must be valid JSON", 400)

    uid, err = _int_field(data, "user_id")
    if err: return _err(err, 422)

    top_k, err = _int_field(data, "top_k", default=DEFAULT_TOP_K, min_val=1, max_val=20)
    if err: return _err(err, 422)

    user_gender = user_gender_map.get(uid)
    profile     = _build_profile(uid)

    # Cold start — new user with no history
    if profile["_source"] == "cold-start":
        results = _popular_for_gender(user_gender, topk=top_k)
        return jsonify({
            "user_id":         uid,
            "user_gender":     user_gender,
            "profile_source":  "cold-start",
            "recommendations": results,
        }), 200

    # Existing user — hybrid ranking
    live_pids   = [e["product_id"] for e in interaction_store.get(uid, [])]
    exclude_ids = profile["purchased_ids"]

    candidates = _get_candidates(
        history     = list(set(live_pids)),
        user_gender = user_gender,
        exclude_ids = exclude_ids,
        topk        = CANDIDATE_POOL,
    )

    if not candidates:
        return _err("No eligible candidates found", 404)

    results = _rank_hybrid(candidates, profile, uid, topk=top_k)

    if not results:
        return _err("Ranking produced no results", 500)

    return jsonify({
        "user_id":         uid,
        "user_gender":     user_gender,
        "profile_source":  profile["_source"],
        "recommendations": results,
    }), 200


# GET /user/<user_id>
@app.route("/user/<int:uid>", methods=["GET"])
def get_user(uid):
    """Inspect the current profile of a user — useful for debugging."""
    profile = _build_profile(uid)
    _refresh_decay(uid)

    try:
        fav_sub    = subcat_enc.inverse_transform([int(profile["fav_subcategory"])])[0]
        fav_gender = gender_enc.inverse_transform([int(profile["fav_gender"])])[0]
        fav_season = season_enc.inverse_transform([int(profile["fav_season"])])[0]
    except Exception:
        fav_sub = fav_gender = fav_season = "unknown"

    live_events = [
        {
            "product_id": e["product_id"],
            "event_type": e["event_type"],
            "timestamp":  e["timestamp"].isoformat(),
            "event_w":    round(e["event_w"],   4),
            "decay_w":    round(e["decay_w"],   4),
            "combined_w": round(e["combined_w"],4),
        }
        for e in interaction_store.get(uid, [])
    ]

    return jsonify({
        "user_id":          uid,
        "registered_gender": user_gender_map.get(uid),
        "profile_source":   profile["_source"],
        "fav_subcategory":  fav_sub,
        "fav_gender":       fav_gender,
        "fav_season":       fav_season,
        "fav_price_tier":   profile["fav_price_tier"],
        "avg_price":        round(profile["avg_price"], 2),
        "purchase_count":   profile["purchase_count"],
        "live_interactions": live_events,
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
