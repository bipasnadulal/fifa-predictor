import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    team_stats      = pd.read_csv('data/team_stats.csv')
    group_preds     = pd.read_csv('outputs/group_stage_predictions.csv')
    group_standings = pd.read_csv('outputs/group_standings.csv')
    knockout_preds  = pd.read_csv('outputs/knockout_predictions.csv')
    return team_stats, group_preds, group_standings, knockout_preds

team_stats, group_preds, group_standings, knockout_preds = load_data()

# ── Predict function ──────────────────────────────────────
def predict_match(home_team, away_team, max_goals=6):
    home = team_stats[team_stats['team'] == home_team].iloc[0]
    away = team_stats[team_stats['team'] == away_team].iloc[0]

    home_xg = np.clip(
        home['home_goals_scored'] * away['away_goals_conceded'] / 0.9, 0.3, 4.0)
    away_xg = np.clip(
        away['away_goals_scored'] * home['home_goals_conceded'] / 1.1, 0.3, 4.0)

    home_probs   = [poisson.pmf(i, home_xg) for i in range(max_goals+1)]
    away_probs   = [poisson.pmf(i, away_xg) for i in range(max_goals+1)]
    score_matrix = np.outer(home_probs, away_probs)

    home_win = np.tril(score_matrix, -1).sum()
    draw     = np.trace(score_matrix)
    away_win = np.triu(score_matrix, 1).sum()

    max_idx  = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

    return {
        'home_xg': round(home_xg, 2),
        'away_xg': round(away_xg, 2),
        'home_goals': int(max_idx[0]),
        'away_goals': int(max_idx[1]),
        'home_win': round(home_win * 100, 1),
        'draw': round(draw * 100, 1),
        'away_win': round(away_win * 100, 1),
    }

# ── Sidebar navigation ────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/5/5e/FIFA_World_Cup_2026_logo.svg/200px-FIFA_World_Cup_2026_logo.svg.png",
    width=180
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Match Predictor",
    "Group Stage",
    "Knockout Bracket",
    "Team Stats",
])

# ═══════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════
if page == "🏠 Home":
    st.title("FIFA World Cup 2026 Predictor")
    st.markdown("### Built with Python · Poisson Model · Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Teams", "48")
    col2.metric("Matches", "104")
    col3.metric("Host Nations", "3")
    col4.metric("Start Date", "June 11, 2026")

    st.markdown("---")
    st.markdown("### What this app does")
    st.markdown("""
    - **Predicts** every match in the 2026 FIFA World Cup
    - Uses **Poisson regression** trained on 10+ years of international football data
    - Simulates the full **knockout bracket** from Round of 32 to the Final
    - Shows **win probabilities**, expected goals, and score predictions
    - Lets you **test any matchup** with the Match Predictor
    """)

    st.markdown("---")
    # Show predicted winner
    final = knockout_preds[knockout_preds['round'] == 'FINAL']
    if not final.empty:
        winner = final.iloc[0]['winner']
        st.success(f"**Predicted World Cup 2026 Winner: {winner}**")

# ═══════════════════════════════════════════
# PAGE 2 — MATCH PREDICTOR
# ═══════════════════════════════════════════
elif page == "Match Predictor":
    st.title("Match Predictor")
    st.markdown("Pick any two teams and get an instant prediction!")
    st.markdown("---")

    all_teams = sorted(team_stats['team'].tolist())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", all_teams,
                                  index=all_teams.index('Brazil'))
    with col2:
        away_team = st.selectbox("Away Team", all_teams,
                                  index=all_teams.index('Argentina'))

    if home_team == away_team:
        st.warning("Please select two different teams!")
    else:
        pred = predict_match(home_team, away_team)

        st.markdown("---")

        # Score display
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"## {home_team}")
            st.metric("Expected Goals", pred['home_xg'])
        with col2:
            st.markdown(f"## {pred['home_goals']} — {pred['away_goals']}")
            st.markdown("**Predicted Score**")
        with col3:
            st.markdown(f"## {away_team} ✈️")
            st.metric("Expected Goals", pred['away_xg'])

        st.markdown("---")

        # Win probability bar
        st.markdown("### Win Probabilities")
        fig = go.Figure(go.Bar(
            x=[pred['home_win'], pred['draw'], pred['away_win']],
            y=[home_team, 'Draw', away_team],
            orientation='h',
            marker_color=['#1f77b4', '#aaaaaa', '#ff7f0e'],
            text=[f"{pred['home_win']}%", f"{pred['draw']}%", f"{pred['away_win']}%"],
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title="Probability (%)",
            height=250,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 3 — GROUP STAGE
# ═══════════════════════════════════════════
elif page == "📊 Group Stage":
    st.title("📊 Group Stage Predictions")
    st.markdown("---")

    selected_group = st.selectbox(
        "Select a Group",
        sorted(group_standings['group'].unique())
    )

    # Standings table
    st.markdown(f"### Group {selected_group} Standings")
    table = group_standings[group_standings['group'] == selected_group][
        ['team', 'played', 'won', 'drawn', 'lost',
         'goals_for', 'goals_against', 'goal_diff', 'points']
    ].reset_index(drop=True)
    table.index += 1
    st.dataframe(table, use_container_width=True)

    # Group matches
    st.markdown(f"### Group {selected_group} Match Results")
    g_matches = group_preds[group_preds['group'] == selected_group][
        ['home_team', 'home_goals', 'away_goals', 'away_team', 'winner']
    ].reset_index(drop=True)
    st.dataframe(g_matches, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 4 — KNOCKOUT BRACKET
# ═══════════════════════════════════════════
elif page == "🏆 Knockout Bracket":
    st.title("🏆 Knockout Stage Predictions")
    st.markdown("---")

    rounds = ['Round of 32', 'Round of 16', 'Quarter-final',
              'Semi-final', 'Third-place playoff', 'FINAL']

    for round_name in rounds:
        round_data = knockout_preds[knockout_preds['round'] == round_name]
        if round_data.empty:
            continue

        st.markdown(f"### {round_name}")
        display = round_data[
            ['home_team', 'home_goals', 'away_goals',
             'away_team', 'winner', 'penalties']
        ].reset_index(drop=True)
        st.dataframe(display, use_container_width=True)
        st.markdown("---")

    # Highlight winner
    final = knockout_preds[knockout_preds['round'] == 'FINAL']
    if not final.empty:
        winner = final.iloc[0]['winner']
        st.balloons()
        st.success(f"🏆 Predicted World Cup 2026 Champion: **{winner}**")

# ═══════════════════════════════════════════
# PAGE 5 — TEAM STATS
# ═══════════════════════════════════════════
elif page == "📈 Team Stats":
    st.title("📈 Team Statistics")
    st.markdown("---")

    # Top teams by win rate
    st.markdown("### Top 15 Teams by Win Rate")
    top15 = team_stats.nlargest(15, 'win_rate')[
        ['team', 'win_rate', 'avg_goals_scored',
         'avg_goals_conceded', 'fifa_rank']
    ]
    fig = px.bar(
        top15,
        x='win_rate', y='team',
        orientation='h',
        color='win_rate',
        color_continuous_scale='Blues',
        labels={'win_rate': 'Win Rate', 'team': 'Team'}
    )
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Full stats table
    st.markdown("### Full Team Stats Table")
    st.dataframe(
        team_stats.sort_values('win_rate', ascending=False).reset_index(drop=True),
        use_container_width=True
    )