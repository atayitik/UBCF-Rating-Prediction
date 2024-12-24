#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <stdexcept>
using namespace std;

class RecommenderSystem {
private:
    vector<vector<float>> ratings;
    int numUsers;
    int numMovies;

    float cosineSimilarity(const vector<float>& userA, const vector<float>& userB) {
        float dotProduct = 0.0, magnitudeA = 0.0, magnitudeB = 0.0;
        for (size_t i = 0; i < userA.size(); ++i) {
            if (userA[i] != 0 && userB[i] != 0) {
                dotProduct += userA[i] * userB[i];
                magnitudeA += userA[i] * userA[i];
                magnitudeB += userB[i] * userB[i];
            }
        }
        if (magnitudeA == 0 || magnitudeB == 0) return 0.0;
        return dotProduct / (sqrt(magnitudeA) * sqrt(magnitudeB));
    }

public:
    RecommenderSystem(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }

        string line;
        map<int, map<int, float>> data;

        while (getline(file, line)) {
            stringstream ss(line);
            int userID, movieID;
            float rating;
            char comma;

            ss >> userID >> comma >> movieID >> comma >> rating;
            data[userID][movieID] = rating;
        }

        numUsers = data.rbegin()->first;
        numMovies = 0;
        for (const auto& [user, movies] : data) {
            for (const auto& [movie, rating] : movies) {
                numMovies = max(numMovies, movie);
            }
        }

        ratings = vector<vector<float>>(numUsers, vector<float>(numMovies, 0.0));
        for (const auto& [user, movies] : data) {
            for (const auto& [movie, rating] : movies) {
                ratings[user - 1][movie - 1] = rating;
            }
        }
    }

    float predictRating(int targetUser, int targetMovie, int k) {
        if (targetUser < 1 || targetUser > numUsers || targetMovie < 1 || targetMovie > numMovies) {
            throw out_of_range("Invalid user or movie ID");
        }

        vector<pair<float, int>> similarities;

        for (size_t i = 0; i < ratings.size(); ++i) {
            if ((int)i != targetUser - 1 && ratings[i][targetMovie - 1] != 0) {
                float similarity = cosineSimilarity(ratings[targetUser - 1], ratings[i]);
                similarities.push_back({similarity, (int)i});
            }
        }

        sort(similarities.rbegin(), similarities.rend());

        float numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < min(k, (int)similarities.size()); ++i) {
            int similarUser = similarities[i].second;
            float similarity = similarities[i].first;
            numerator += similarity * ratings[similarUser][targetMovie - 1];
            denominator += fabs(similarity);
        }

        return denominator == 0 ? 0 : numerator / denominator;
    }

    void displayStats() const {
        cout << "Number of Users: " << numUsers << endl;
        cout << "Number of Movies: " << numMovies << endl;
    }
};

int main() {
    try {
        string filename = "training_data.csv";
        RecommenderSystem recommender(filename);

        recommender.displayStats();

        int targetUser = 1;
        int targetMovie = 2;
        int k = 5;

        float predictedRating = recommender.predictRating(targetUser, targetMovie, k);
        cout << "Predicted Rating for User " << targetUser << " on Movie " << targetMovie << ": " << predictedRating << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
