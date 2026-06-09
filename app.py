import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="FIFA 2026 Predictor",
    layout="centered"
)

# Load data and models 
@st.cache_data
def load_data():
    team_stats      = pd.read_csv('data/team_stats.csv')
    group_standings = pd.read_csv('outputs/group_standings.csv')
    knockout_preds  = pd.read_csv('outputs/knockout_predictions.csv')
    return team_stats, group_standings, knockout_preds

@st.cache_resource
def load_models():
    outcome_model    = joblib.load('src/outcome_model.pkl')
    home_goals_model = joblib.load('src/home_goals_model.pkl')
    away_goals_model = joblib.load('src/away_goals_model.pkl')
    return outcome_model, home_goals_model, away_goals_model

team_stats, group_standings, knockout_preds = load_data()
outcome_model, home_goals_model, away_goals_model = load_models()

# Predict function
def predict_match(home_team, away_team):
    home = team_stats[team_stats['team'] == home_team].iloc[0]
    away = team_stats[team_stats['team'] == away_team].iloc[0]

    features = pd.DataFrame([{
        'home_goals_scored_avg':   home['avg_goal_scored'],
        'home_goals_conceded_avg': home['avg_goal_conceded'],
        'home_win_rate':           home['win_rate'],
        'home_draw_rate':          home['draw_rate'],
        'home_form_points':        home.get('form_points', 1.0),
        'home_rank':               home['fifa_rank'],
        'away_goals_scored_avg':   away['avg_goal_scored'],
        'away_goals_conceded_avg': away['avg_goal_conceded'],
        'away_win_rate':           away['win_rate'],
        'away_draw_rate':          away['draw_rate'],
        'away_form_points':        away.get('form_points', 1.0),
        'away_rank':               away['fifa_rank'],
        'rank_diff':               away['fifa_rank'] - home['fifa_rank'],
        'goal_avg_diff':           home['avg_goal_scored'] - away['avg_goal_scored'],
        'form_diff':               home.get('form_points', 1.0) - away.get('form_points', 1.0),
    }])

    probs      = outcome_model.predict_proba(features)[0]
    away_win_p = round(probs[0] * 100, 1)
    draw_p     = round(probs[1] * 100, 1)
    home_win_p = round(probs[2] * 100, 1)

    home_goals = int(max(0, round(home_goals_model.predict(features)[0])))
    away_goals = int(max(0, round(away_goals_model.predict(features)[0])))

    return {
        'home_goals':  home_goals,
        'away_goals':  away_goals,
        'home_win_p':  home_win_p,
        'draw_p':      draw_p,
        'away_win_p':  away_win_p,
    }

# Header
st.title("FIFA World Cup 2026 Predictor")
st.caption("Built with Python · XGBoost · Historical match data")
st.divider()

# Predicted Winner
final = knockout_preds[knockout_preds['round'] == 'FINAL']
if not final.empty:
    winner = final.iloc[0]['winner']
    runner = final.iloc[0]['home_team'] if final.iloc[0]['winner'] == final.iloc[0]['away_team'] else final.iloc[0]['away_team']
    st.success(f"Predicted Champion: **{winner}**")

st.divider()

# Match Predictor
st.subheader("Predict a Match")

all_teams = sorted(team_stats['team'].tolist())
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", all_teams,
                              index=all_teams.index('Brazil'))
with col2:
    away_team = st.selectbox("Away Team", all_teams,
                              index=all_teams.index('Argentina'))

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    pred = predict_match(home_team, away_team)

    # Score
    c1, c2, c3 = st.columns([2, 1, 2])
    c1.metric(home_team, pred['home_goals'])
    c2.markdown("<h3 style='text-align:center; padding-top:20px'>vs</h3>",
                unsafe_allow_html=True)
    c3.metric(away_team, pred['away_goals'])

    # Probability bar
    fig = go.Figure(go.Bar(
        x=[pred['home_win_p'], pred['draw_p'], pred['away_win_p']],
        y=[home_team, 'Draw', away_team],
        orientation='h',
        marker_color=['#2ecc71', '#95a5a6', '#e74c3c'],
        text=[f"{pred['home_win_p']}%",
              f"{pred['draw_p']}%",
              f"{pred['away_win_p']}%"],
        textposition='auto'
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

#  Group Standings
st.subheader("Group Standings")

selected_group = st.selectbox(
    "Select Group",
    sorted(group_standings['group'].unique())
)

table = group_standings[group_standings['group'] == selected_group][
    ['team', 'played', 'won', 'drawn', 'lost', 'goals_for', 'goal_diff', 'points']
].reset_index(drop=True)
table.index += 1
st.dataframe(table, use_container_width=True, hide_index=False)

st.divider()

# Knockout Results
st.subheader("Knockout Results")

rounds = ['Round of 32', 'Round of 16', 'Quarter-final',
          'Semi-final', 'Third-place playoff', 'FINAL']

selected_round = st.selectbox("Select Round", rounds)

round_data = knockout_preds[knockout_preds['round'] == selected_round][
    ['home_team', 'home_goals', 'away_goals', 'away_team', 'winner', 'penalties']
].reset_index(drop=True)

st.dataframe(round_data, use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: Kaggle international football results · Rankings: FIFA · Model: XGBoost")