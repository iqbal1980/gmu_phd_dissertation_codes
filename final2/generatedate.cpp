#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <thread>
#include <mutex>


/*
 159  g++
  160  g++ -o test.cpp
  161  g++ -o test test.cpp
  169  g++ -O3 -march=native -mtune=native -o test test.cpp
  177  g++ -O3 -march=native -mtune=native -std=c++11 -pthread -o test test.cpp
*/

using namespace std;

// Constants for safe_log and safe_exponential
const double MIN_VALUE = -80;  // Minimum physiologically reasonable value for Vm
const double MAX_VALUE = 40;   // Maximum physiologically reasonable value for Vm

// Function for safe_log
double safe_log(double x) {
    if (x <= 0)
        return MIN_VALUE;
    return log(x);
}

// Function for exponential_function
double exponential_function(double x, double a) {
    return exp(a * x);
}

// Simulate process function
vector<double> simulate_process_modified_v2(double g_gap_value, double Ibg_init, double Ikir_coef, double cm, double dx, double K_o, vector<double>& I_app) {
    double dt = 0.001;
    double F = 9.6485e4;
    double R = 8.314e3;
    int loop = 600000;
    int Ng = 200;
    vector<double> Vm(Ng, -33);
    double g_gap = g_gap_value;
    double eki1 = (g_gap * dt) / (dx * dx * cm);
    double eki2 = dt / cm;

    vector<double> I_bg(Ng, Ibg_init);
    vector<double> I_kir(Ng, 0);

    for (int j = 0; j < loop; j++) {
        double t = j * dt;
        if (100 <= t && t <= 400)
            I_app[99] = I_app[99];
        else
            I_app[99] = 0.0;

        for (int kk = 0; kk < Ng; kk++) {
            double E_K = (R * 293 / F) * safe_log(K_o / 150);
            I_bg[kk] = Ibg_init * (Vm[kk] + 30);
            I_kir[kk] = Ikir_coef * sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)));

            double new_Vm = Vm[kk];
            if (kk == 0)
                new_Vm += 3 * (Vm[kk + 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk]);
            else if (kk == Ng - 1)
                new_Vm += eki1 * (Vm[kk - 1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk]);
            else if (kk == 98 || kk == 99 || kk == 100)
                new_Vm += eki1 * 0.6 * (Vm[kk + 1] + Vm[kk - 1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk]);
            else
                new_Vm += eki1 * (Vm[kk + 1] + Vm[kk - 1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk]);

            // Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VALUE), MIN_VALUE);
        }
    }

    return Vm;
}

// Function to generate and write training data for a range of samples
void generate_training_data_range(int start_index, int end_index, const vector<pair<double, double>>& param_ranges, const string& output_file, int& lines_written, mutex& mtx) {
    ofstream file(output_file, ios::app);
    if (!file.is_open()) {
        cout << "Failed to open file: " << output_file << endl;
        return;
    }

    random_device rd;
    mt19937 gen(rd());

    for (int i = start_index; i < end_index; i++) {
        vector<double> params;
        for (const auto& range : param_ranges) {
            uniform_real_distribution<> dis(range.first, range.second);
            params.push_back(dis(gen));
        }

        double ggap = params[0];
        double Ikir_coef = params[1];
        double cm = params[2];
        double K_o = params[3];
        double I_app_value = params[4];

        double dx = 1;
        double Ibg_init = 0.7 * 0.94;

        vector<double> I_app_array(200, 0);
        I_app_array[99] = I_app_value;

        vector<double> Vm = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_array);

        // Write parameter values
        for (int j = 0; j < params.size(); j++) {
            file << params[j];
            if (j < params.size() - 1)
                file << ",";
        }

        // Write Vm values
        for (int j = 0; j < Vm.size(); j++) {
            file << "," << Vm[j];
        }
        file << endl;

        // Increment lines_written and log progress
        {
            lock_guard<mutex> lock(mtx);
            lines_written++;
            if (lines_written % 10 == 0)
                cout << "Generated " << lines_written << " lines..." << endl;
        }
    }

    file.close();
}

// Generate training data using multiple threads
void generate_training_data(int num_samples, const string& output_file, int num_threads) {
    auto start_time = chrono::high_resolution_clock::now();

    vector<pair<double, double>> param_ranges = {
        {0.1, 35},    // ggap
        {0.5, 0.96}, // Ikir_coef
        {6, 15},      // cm
        {1, 8},       // K_o
        {-100, 100}     // I_app
    };

    ofstream file(output_file);
    if (!file.is_open()) {
        cout << "Failed to open file: " << output_file << endl;
        return;
    }

    // Write header row
    file << "ggap,Ikir_coef,cm,K_o,I_app";
    for (int i = 0; i < 200; i++)
        file << ",Vm_" << i;
    file << endl;
    file.close();

    int lines_written = 0;
    mutex mtx;

    vector<thread> threads;
    int samples_per_thread = num_samples / num_threads;
    int remaining_samples = num_samples % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_index = i * samples_per_thread;
        int end_index = (i + 1) * samples_per_thread;
        if (i == num_threads - 1)
            end_index += remaining_samples;

        threads.emplace_back(generate_training_data_range, start_index, end_index, ref(param_ranges), ref(output_file), ref(lines_written), ref(mtx));
    }

    for (auto& t : threads)
        t.join();

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
    cout << "Generating " << num_samples << " samples took " << duration << " seconds." << endl;
}

int main() {
    int num_samples = 10000;
    string output_file = "training_data.csv";
    int num_threads = thread::hardware_concurrency();  // Use the maximum number of hardware threads
	printf("Max number of threads is = %d\n", num_threads);
    generate_training_data(num_samples, output_file, num_threads);

    return 0;
}

