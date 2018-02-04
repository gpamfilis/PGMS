class Elo2(object):

    def match(self, p1, p2):
        return self.match_algo_strict(p1, p2)

    @staticmethod
    def match_algo_strict(previous_winner_score, previous_looser_score):
        r1 = max(min(previous_looser_score - previous_winner_score, 400), -400)
        r2 = max(min(previous_winner_score - previous_looser_score, 400), -400)
        e1 = 1.0 / (1+10**(r1 / 400))
        e2 = 1.0 / (1+10**(r2 / 400))
        s1 = 1
        s2 = 0
        new_winner_score = previous_winner_score + K*(s1-e1)
        new_looser_score = previous_looser_score + K*(s2-e2)

        return new_winner_score, new_looser_score



def compute_elo_by_goals2(data_df, players, all_teams, elo=Elo2(), initial_score=100):
    """
    This function is used to compute the elo ratings of teams based on goals scored depending on wins and losses.

    :param data_df:
    :param players:
    :param elo:
    :return:
    """

    n_games = data_df.shape[0]
    # does this work for sparce matrix
    elo_table = lil_matrix((n_games, len(players)))

    # elo_table = np.zeros((n_games, len(players)))
    # set initial score for all teams
    elo_table[0, :] = initial_score
    bar = progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ])
    for i in bar(range(1, n_games)):

        match = data_df.iloc[i-1]
        player_home = match.home_team
        player_away = match.away_team
        hid = np.where(all_teams == player_home)[0][0]
        aid = np.where(all_teams == player_away)[0][0]
        pair = [elo_table[i-1, hid], elo_table[i-1, aid]]
        res = match.result_final
        home_goals = match.home_goals
        away_goals = match.away_goals

        if res == 0:
            for goal_difference in range(int(abs(home_goals-away_goals))):
                a, b = elo.match_algo_strict(pair[0], pair[1])
            elo_table[i, hid] = a
            elo_table[i, aid] = b
        elif res == 2:
            for goal_difference in range(int(abs(home_goals-away_goals))):
                a, b = elo.match(pair[1], pair[0])
            elo_table[i, aid] = a
            elo_table[i, hid] = b
        else:
            pass
    return elo_table
