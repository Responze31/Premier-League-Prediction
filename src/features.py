import pandas as pd
import numpy as np

def get_res(r):
    """0=home win, 1=away win, 2=draw"""
    if r.hg > r.ag:
        return 0
    elif r.hg < r.ag:
        return 1
    else:
        return 2

def teams_form_enhanced(df, w=5):
    """Calculate rolling home/away form for each team"""
    df = df.sort_values('date').reset_index(drop=True)

    teams = pd.unique(df[["home", "away"]].values.ravel())
    home_hist = {t: [] for t in teams}
    away_hist = {t: [] for t in teams}
    rows = []

    for _, r in df.iterrows():
        h, a = r.home, r.away

        hf_home = home_hist[h][-w:]
        af_away = away_hist[a][-w:]

        rows.append({
            "h_gf": np.mean([x[0] for x in hf_home]) if hf_home else 1.5,
            "h_ga": np.mean([x[1] for x in hf_home]) if hf_home else 1.0,
            "a_gf": np.mean([x[0] for x in af_away]) if af_away else 1.0,
            "a_ga": np.mean([x[1] for x in af_away]) if af_away else 1.5,
        })

        home_hist[h].append((r.hg, r.ag))
        away_hist[a].append((r.ag, r.hg))

    return pd.concat([df, pd.DataFrame(rows)], axis=1)

def add_team_strength(df, historical_df):
    """Add PPG and goal difference features"""
    team_stats = {}

    for team in pd.unique(historical_df[["home", "away"]].values.ravel()):
        pts, gf, ga, games = 0, 0, 0, 0

        home = historical_df[historical_df['home'] == team]
        for _, r in home.iterrows():
            games += 1
            gf += r.hg
            ga += r.ag
            if r.hg > r.ag: pts += 3
            elif r.hg == r.ag: pts += 1

        away = historical_df[historical_df['away'] == team]
        for _, r in away.iterrows():
            games += 1
            gf += r.ag
            ga += r.hg
            if r.ag > r.hg: pts += 3
            elif r.ag == r.hg: pts += 1

        ppg = pts / max(games, 1)
        gd_per_game = (gf - ga) / max(games, 1)
        team_stats[team] = {'ppg': ppg, 'gd': gd_per_game}

    df['h_strength'] = df['home'].map(lambda x: team_stats.get(x, {'ppg': 1.0})['ppg']).fillna(1.0)
    df['a_strength'] = df['away'].map(lambda x: team_stats.get(x, {'ppg': 1.0})['ppg']).fillna(1.0)
    df['h_gd'] = df['home'].map(lambda x: team_stats.get(x, {'gd': 0})['gd']).fillna(0)
    df['a_gd'] = df['away'].map(lambda x: team_stats.get(x, {'gd': 0})['gd']).fillna(0)

    return df
