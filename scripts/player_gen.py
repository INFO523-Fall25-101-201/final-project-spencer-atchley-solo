import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------
# Load game data
# ---------------------------
reviews = pd.read_csv("../data/video_game_reviews.csv")

# Clean up / ensure categories
reviews = reviews.dropna(subset=["Game Title", "Genre", "Platform"])

# For convenience, rename columns to simpler names
reviews = reviews.rename(
    columns={
        "Game Title": "game_title",
        "Age Group Targeted": "age_group_targeted",
        "Requires Special Device": "requires_special_device",
        "Release Year": "release_year",
        "Multiplayer": "multiplayer",
        "Game Mode": "game_mode",
        "Price": "price",
        "User Rating": "base_user_rating",
    }
)

# Categorical distributions from the dataset
genre_probs = reviews["Genre"].value_counts(normalize=True)
platform_probs = reviews["Platform"].value_counts(normalize=True)
age_probs = reviews["age_group_targeted"].value_counts(normalize=True)
mode_probs = reviews["game_mode"].value_counts(normalize=True)
multi_probs = reviews["multiplayer"].value_counts(normalize=True)
special_probs = reviews["requires_special_device"].value_counts(normalize=True)

# Price quantiles to define cheap / mid / expensive
price = reviews["price"].dropna()
low_q, high_q = price.quantile([0.33, 0.66])


def weighted_choice(vc_series, size=1):
    """Sample from a value_counts() series (index = values, values = probs)."""
    return rng.choice(vc_series.index.to_list(), size=size, p=vc_series.values)


def sample_price_sensitivity(n):
    # Rough: 40% budget, 40% midrange, 20% premium
    return rng.choice(["Budget", "Midrange", "Premium"], size=n, p=[0.4, 0.4, 0.2])


# ---------------------------
# Generate players
# ---------------------------
n_users = 1000  # tweak as needed

user_ids = np.arange(1, n_users + 1)

# Give users their own age group (loosely based on what games target)
user_age_group = weighted_choice(age_probs, size=n_users)

# Sessions per week (casual to addict)
sessions_per_week = []
for _ in range(n_users):
    if rng.random() < 0.75:
        sessions_per_week.append(rng.integers(2, 7))  # 2–6
    else:
        sessions_per_week.append(rng.integers(7, 15))  # 7–14
sessions_per_week = np.array(sessions_per_week)

# Average session length (hours)
avg_session_length_hours = np.clip(
    rng.normal(loc=2.0, scale=0.7, size=n_users), 0.5, 6.0
)

players = pd.DataFrame(
    {
        "user_id": user_ids,
        "age_group": user_age_group,
        "session_count_per_week": sessions_per_week,
        "avg_session_length_hours": avg_session_length_hours.round(2),
    }
)

players.to_csv("../data/players.csv", index=False)
print("Generated players.csv with", len(players), "users")

# ---------------------------
# Generate player game libraries
# ---------------------------

player_game_rows = []

for uid in user_ids:
    # ---- latent preferences for this user (NOT stored directly) ----
    primary_genre = weighted_choice(genre_probs, size=1)[0]
    # secondary genre: 30% same, else different
    if rng.random() < 0.3:
        secondary_genre = primary_genre
    else:
        other_genres = [g for g in genre_probs.index if g != primary_genre]
        secondary_genre = rng.choice(other_genres)

    preferred_platform = weighted_choice(platform_probs, size=1)[0]
    prefers_multiplayer = weighted_choice(multi_probs, size=1)[0]
    preferred_mode = weighted_choice(mode_probs, size=1)[0]
    prefers_special_device = weighted_choice(special_probs, size=1)[0]
    price_sensitivity = sample_price_sensitivity(1)[0]

    # Number of games owned (10–60)
    n_games_owned = int(rng.integers(10, 61))

    # Compute a per-game preference score
    g = reviews.copy()

    # Start with all ones
    score = np.ones(len(g), dtype=float)

    # Genre preference
    score *= np.where(g["Genre"] == primary_genre, 3.0, 1.0)
    score *= np.where(g["Genre"] == secondary_genre, 2.0, 1.0)

    # Platform preference
    score *= np.where(g["Platform"] == preferred_platform, 2.0, 1.0)

    # Multiplayer preference
    score *= np.where(g["multiplayer"] == prefers_multiplayer, 1.5, 1.0)

    # Special device preference
    if prefers_special_device == "Yes":
        score *= np.where(g["requires_special_device"] == "Yes", 1.5, 1.0)
    else:
        # Small penalty if game requires special device but user doesn't like that
        score *= np.where(g["requires_special_device"] == "Yes", 0.7, 1.0)

    # Price sensitivity
    if price_sensitivity == "Budget":
        score *= np.where(g["price"] <= low_q, 1.5, 1.0)
        score *= np.where(g["price"] > high_q, 0.6, 1.0)
    elif price_sensitivity == "Premium":
        score *= np.where(g["price"] >= high_q, 1.3, 1.0)

    # Avoid all-zeros
    score = np.clip(score, 1e-6, None)

    # Convert to probabilities
    probs = score / score.sum()

    # Sample games without replacement according to preference probabilities
    owned_indices = rng.choice(
        len(g), size=min(n_games_owned, len(g)), replace=False, p=probs
    )
    owned_games = g.iloc[owned_indices].copy()

    # Assign playtime based on preference score (more preferred -> more hours)
    owned_scores = score[owned_indices]
    # Normalize owned_scores to 0.5–1.5 multiplier
    owned_scores_norm = (owned_scores - owned_scores.min()) / (
        owned_scores.max() - owned_scores.min() + 1e-9
    )
    multipliers = 0.5 + owned_scores_norm  # 0.5–1.5

    base_hours = rng.normal(loc=10, scale=5, size=len(owned_games))  # avg 10h, noisy
    playtime_hours = np.clip(base_hours * multipliers, 0.5, 500)

    # Synthetic player rating (1–5) based on preference + noise
    rating_base = 3.0 + 1.0 * (owned_scores_norm - 0.5)  # around 2.5–3.5
    rating_noise = rng.normal(loc=0.0, scale=0.7, size=len(owned_games))
    player_rating = np.clip(rating_base + rating_noise, 1.0, 5.0)

    for idx, row in owned_games.iterrows():
        player_game_rows.append(
            {
                "user_id": uid,
                "game_title": row["game_title"],
                "platform": row["Platform"],
                "genre": row["Genre"],
                "multiplayer": row["multiplayer"],
                "requires_special_device": row["requires_special_device"],
                "price": row["price"],
                "playtime_hours": float(playtime_hours[owned_games.index.get_loc(idx)]),
                "player_rating": round(
                    float(player_rating[owned_games.index.get_loc(idx)]), 2
                ),
            }
        )

player_games = pd.DataFrame(player_game_rows)
player_games.to_csv("../data/player_games.csv", index=False)
print("Generated player_games.csv with", len(player_games), "rows")
