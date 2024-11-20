
from flask import Flask, render_template, request

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)


# Sample datasets with realistic cricket player data
# Columns: [Batting Average, Bowling Average, Strike Rate, Economy Rate]

# Bowlers Data
bowlers_data = np.array([
    [25.4, 22.3, 4.5, 3.8],
    [20.1, 18.2, 4.2, 3.5],
    [28.5, 25.0, 3.8, 4.2],
    [23.8, 21.5, 4.0, 3.7],
    [26.7, 24.0, 4.2, 4.0],
    [22.3, 20.8, 4.1, 3.9],
    [24.5, 23.2, 4.3, 3.8],
    [21.0, 19.8, 3.9, 3.6],
    [27.2, 24.5, 4.4, 4.1],
    [22.8, 21.0, 4.1, 3.8],
    [26.1, 23.5, 4.3, 4.0],
    [24.0, 22.2, 4.2, 3.7],
    [23.5, 21.7, 4.1, 3.9],
    [25.8, 24.3, 4.4, 4.2],
    [20.9, 19.5, 3.8, 3.5]
])

# Batsmen Data
batsman_data = np.array([
    [45.6, 0, 78.2, 0],
    [38.7, 0, 85.5, 0],
    [42.1, 0, 80.8, 0],
    [48.3, 0, 75.6, 0],
    [40.5, 0, 82.3, 0],
    [46.8, 0, 79.5, 0],
    [43.2, 0, 83.1, 0],
    [41.7, 0, 81.4, 0],
    [47.5, 0, 76.7, 0],
    [39.4, 0, 84.2, 0],
    [44.2, 0, 79.8, 0],
    [42.8, 0, 82.0, 0],
    [46.1, 0, 77.3, 0],
    [38.9, 0, 85.2, 0],
    [43.7, 0, 81.9, 0]
])

# All-Rounders Data
allrounders_data = np.array([
    [35.4, 18.2, 72.3, 4.2],
    [40.8, 20.5, 76.7, 4.5],
    [37.2, 22.0, 70.5, 4.0],
    [38.5, 19.8, 73.2, 4.1],
    [36.7, 21.3, 71.8, 4.3],
    [39.2, 20.0, 75.1, 4.4],
    [37.9, 21.8, 72.8, 4.2],
    [40.1, 19.5, 74.6, 4.3],
    [36.0, 22.5, 71.0, 4.1],
    [38.8, 20.2, 73.7, 4.4],
    [37.5, 21.0, 72.2, 4.2],
    [39.7, 19.3, 75.9, 4.5],
    [38.0, 20.8, 74.3, 4.2],
    [36.5, 22.2, 71.5, 4.0],
    [39.0, 19.6, 76.3, 4.4]
])



bowlers_labels = np.random.rand(15, 1) * 100
batsman_labels = np.random.rand(15, 1) * 100
allrounders_labels = np.random.rand(15, 1) * 100

bowlers_names = ["Bumraah", "M_Sahammi", "M_Siraj", "S_Thakur", "D_Chahar", "A_Madwaal", "U_Yadav", "Saini", "Harshdeep", "Yuvi", "K_yadav", "R_Ashwin", "Haizal", "M_Stark", "P_Cummins"]
batsman_names = ["Rohit", "Virat", "shubman", "shreyas", "surykumar", "KL_Rahul", "Ishan", "N_Tilak", "Nehal_Varma", "Tim_David", "Vishnu", "Suresh_Rana", "MS_Dhoni", "Sachin_Tedulkar", "Gatum"]
allrounders_names = ["Ravindra", "Hardik", "Axar", "Krunal", "Pollard", "Cameron_green", "Dube", "Vijay_Shankar", "Rahul_Tewatia", "Rutik_jagtap", "Harshal_Patel", "Arjun_Tedulkar", "Romario_Shephrd", "Kumar_Kartikeya", "Shams_Mulani"]

team1_batsman = batsman_names[:7]
team2_batsman = batsman_names[8:14]
team1_bowlers = bowlers_names[:7]
team2_bowlers = bowlers_names[8:14]
team1_allrounders = allrounders_names[:5]
team2_allrounders = allrounders_names[6:10]
#num_players = int(input("Enter the number of players you want in the recommendation: "))

max_batsman = 5
max_bowlers = 4
max_allrounders = 2

team1_data = np.concatenate(
    (bowlers_data[:5], batsman_data[:6], allrounders_data[:4]),
    axis=0
)
team2_data = np.concatenate(
    (bowlers_data[5:10], batsman_data[6:12], allrounders_data[4:8]),
    axis=0
)

scaler = StandardScaler()
team1_data = scaler.fit_transform(team1_data)
team2_data = scaler.transform(team2_data)

X_train_team1, X_test_team1, y_train_team1, y_test_team1 = train_test_split(
    team1_data, np.concatenate((bowlers_labels[:5], batsman_labels[:6], allrounders_labels[:4]), axis=0),
    test_size=0.2, random_state=42
)

model_team1 = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=1000, random_state=42)
model_team1.fit(X_train_team1, y_train_team1.ravel())

X_train_team2, X_test_team2, y_train_team2, y_test_team2 = train_test_split(
    team2_data, np.concatenate((bowlers_labels[5:10], batsman_labels[6:12], allrounders_labels[4:8]), axis=0),
    test_size=0.2, random_state=42
)

model_team2 = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=1000, random_state=42)
model_team2.fit(X_train_team2, y_train_team2.ravel())

selected_players_team1 = []
selected_players_team2 = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        my_team_name = request.form['team']
        opponent_team_name = request.form['opponent']
        required_player_type = request.form['player_type']
        match_type = request.form['match_type']

        if my_team_name.lower() == "team1":
            model = model_team1
            selected_players = selected_players_team1
            if required_player_type.lower() == "bowler":
                player_names = [name for name in team1_bowlers if name not in selected_players]
                player_data = bowlers_data[:5]
            elif required_player_type.lower() == "batsman":
                player_names = [name for name in team1_batsman if name not in selected_players]
                player_data = batsman_data[:6]
            elif required_player_type.lower() == "allrounder":
                player_names = [name for name in team1_allrounders if name not in selected_players]
                player_data = allrounders_data[:4]
            else:
                return render_template('error.html', message="Invalid player type. Please enter Bowler, Batsman, "
                                                             "or All Rounder.")
        elif my_team_name.lower() == "team2":
            model = model_team2
            selected_players = selected_players_team2
            if required_player_type.lower() == "bowler":
                player_names = [name for name in team2_bowlers if name not in selected_players]
                player_data = bowlers_data[5:10]
            elif required_player_type.lower() == "batsman":
                player_names = [name for name in team2_batsman if name not in selected_players]
                player_data = batsman_data[6:12]
            elif required_player_type.lower() == "allrounder":
                player_names = [name for name in team1_allrounders + team2_allrounders if name not in selected_players]
                if my_team_name.lower() == "team1":
                    player_data = allrounders_data[:4]
                elif my_team_name.lower() == "team2":
                    player_data = allrounders_data[4:8]
                else:
                    return render_template('error.html', message="Invalid team name. Please enter Team1 or Team2.")
        else:
            return render_template('error.html', message="Invalid team name. Please enter Team1 or Team2.")

        user_input_data = scaler.transform(player_data)
        predicted_performance = model.predict(user_input_data)
        eligible_players = [name for name in player_names if name not in selected_players]
        num_players = int(request.form['num_players'])
        num_to_recommend = min(len(eligible_players), num_players)

        recommendations = []
        for _ in range(num_to_recommend):
            best_player_index = np.argmax(predicted_performance)
            best_player = eligible_players[best_player_index]
            predicted_performance_percentage = predicted_performance[best_player_index]

            recommendations.append({
                "player_name": best_player,
                "predicted_performance": predicted_performance_percentage
            })

            selected_players.append(best_player)
            predicted_performance[best_player_index] = -1

        return render_template('index.html',
                               team1_bowlers=team1_bowlers,
                               team2_bowlers=team2_bowlers,
                               team1_batsmen=team1_batsman,
                               team2_batsmen=team2_batsman,
                               team1_allrounders=team1_allrounders,
                               team2_allrounders=team2_allrounders,
                               selected_players=selected_players,
                               recommendations=recommendations,
                               num_players=num_players)
    else:
        return render_template('index.html',
                               team1_bowlers=team1_bowlers,
                               team2_bowlers=team2_bowlers,
                               team1_batsmen=team1_batsman,
                               team2_batsmen=team2_batsman,
                               team1_allrounders=team1_allrounders,
                               team2_allrounders=team2_allrounders)


if __name__ == '__main__':
    app.run(debug=True)


