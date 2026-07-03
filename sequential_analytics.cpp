#include <random>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <map>

struct AnalyticsResults {
    // Task 1: Basic Statistics
    double min_value;
    double max_value;
    double mean;
    double median;
    double variance;
    double std_deviation;
    long long count;
    
    // Task 2: Histogram
    std::vector<int> histogram;
    int num_bins;
    double bin_width;
    
    // Task 3: Sorting (stores sorted data)
    std::vector<double> sorted_data;
    double sort_time_ms;
    
    // Task 4: Pearson Correlation
    double pearson_correlation;
    
    // Task 5: Moving Average
    std::vector<double> moving_average;
    int window_size;
    
    // Task 6: Outlier Detection
    std::vector<int> outlier_indices;
    std::vector<double> outlier_values;
    int outlier_count;
    double z_score_threshold;
};

// Generate random dataset
std::vector<double> generateDataset(long long numElements) {
    std::vector<double> data;
    data.reserve(numElements);
    
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);
    
    const long long CHUNK_SIZE = 1000000;
    for (long long i = 0; i < numElements; i += CHUNK_SIZE) {
        long long currentChunk = std::min(CHUNK_SIZE, numElements - i);
        for (long long j = 0; j < currentChunk; ++j) {
            data.push_back(dist(rng));
        }
    }
    return data;
}

// TASK 1: Basic Statistics
void computeBasicStats(const std::vector<double>& data, AnalyticsResults& results) {
    results.count = data.size();
    
    if (data.empty()) {
        results.min_value = results.max_value = results.mean = 0;
        results.median = results.variance = results.std_deviation = 0;
        return;
    }
    
    // Min and Max
    results.min_value = *std::min_element(data.begin(), data.end());
    results.max_value = *std::max_element(data.begin(), data.end());
    
    // Mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    results.mean = sum / data.size();
    
    // Median
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    if (data.size() % 2 == 0) {
        results.median = (sorted_data[data.size()/2 - 1] + sorted_data[data.size()/2]) / 2.0;
    } else {
        results.median = sorted_data[data.size()/2];
    }
    
    // Variance and Standard Deviation
    double sum_sq_diff = 0.0;
    for (double val : data) {
        double diff = val - results.mean;
        sum_sq_diff += diff * diff;
    }
    results.variance = sum_sq_diff / data.size();
    results.std_deviation = std::sqrt(results.variance);
}

// TASK 2: Histogram Generation
void computeHistogram(const std::vector<double>& data, AnalyticsResults& results, int numBins = 20) {
    results.num_bins = numBins;
    results.histogram.resize(numBins, 0);
    
    if (data.empty()) return;
    
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    results.bin_width = (maxVal - minVal) / numBins;
    
    for (double val : data) {
        int binIndex = static_cast<int>((val - minVal) / results.bin_width);
        if (binIndex >= numBins) binIndex = numBins - 1;
        results.histogram[binIndex]++;
    }
}

// TASK 3: Sorting (Sequential sort for baseline)
void computeSorting(std::vector<double>& data, AnalyticsResults& results) {
    auto start = std::chrono::high_resolution_clock::now();
    
    results.sorted_data = data;
    std::sort(results.sorted_data.begin(), results.sorted_data.end());
    
    auto end = std::chrono::high_resolution_clock::now();
    results.sort_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
}

// TASK 4: Pearson Correlation (between two halves of the data)
double computePearsonCorrelation(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    
    // Split data into two halves to simulate two variables
    size_t half = data.size() / 2;
    std::vector<double> x(data.begin(), data.begin() + half);
    std::vector<double> y(data.begin() + half, data.end());
    
    // Ensure both have same size
    if (x.size() != y.size()) {
        y.resize(x.size());
    }
    
    size_t n = x.size();
    if (n == 0) return 0.0;
    
    double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
    double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    if (denominator == 0) return 0.0;
    return numerator / denominator;
}

// TASK 5: Moving Average (Rolling Window)
void computeMovingAverage(const std::vector<double>& data, AnalyticsResults& results, int windowSize = 100) {
    results.window_size = windowSize;
    results.moving_average.clear();
    
    if (data.size() < static_cast<size_t>(windowSize)) return;
    
    results.moving_average.reserve(data.size() - windowSize + 1);
    
    // First window sum
    double window_sum = 0.0;
    for (int i = 0; i < windowSize; ++i) {
        window_sum += data[i];
    }
    results.moving_average.push_back(window_sum / windowSize);
    
    // Slide the window
    for (size_t i = windowSize; i < data.size(); ++i) {
        window_sum = window_sum - data[i - windowSize] + data[i];
        results.moving_average.push_back(window_sum / windowSize);
    }
}

// TASK 6: Outlier Detection (Z-score method)
void detectOutliers(const std::vector<double>& data, AnalyticsResults& results, double threshold = 3.0) {
    results.z_score_threshold = threshold;
    results.outlier_indices.clear();
    results.outlier_values.clear();
    
    if (data.empty()) {
        results.outlier_count = 0;
        return;
    }
    
    // Calculate mean and standard deviation
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum_sq_diff = 0.0;
    for (double val : data) {
        double diff = val - mean;
        sum_sq_diff += diff * diff;
    }
    double std_dev = std::sqrt(sum_sq_diff / data.size());
    
    if (std_dev == 0) {
        results.outlier_count = 0;
        return;
    }
    
    // Find outliers using Z-score
    for (size_t i = 0; i < data.size(); ++i) {
        double z_score = std::abs((data[i] - mean) / std_dev);
        if (z_score > threshold) {
            results.outlier_indices.push_back(i);
            results.outlier_values.push_back(data[i]);
        }
    }
    results.outlier_count = results.outlier_indices.size();
}

// Save results to CSV
void saveToCSV(const std::string& filename, const AnalyticsResults& results, 
               const std::vector<double>& timings, double total_time) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }
    
    csv << "Metric,Value\n";
    
    // Task 1: Basic Statistics
    csv << "Count," << results.count << "\n";
    csv << "Min," << std::fixed << std::setprecision(4) << results.min_value << "\n";
    csv << "Max," << results.max_value << "\n";
    csv << "Mean," << results.mean << "\n";
    csv << "Median," << results.median << "\n";
    csv << "Variance," << results.variance << "\n";
    csv << "StdDev," << results.std_deviation << "\n";
    
    // Task 2: Histogram
    csv << "Histogram_Bins," << results.num_bins << "\n";
    csv << "Histogram_BinWidth," << results.bin_width << "\n";
    for (int i = 0; i < results.num_bins && i < 20; ++i) {
        csv << "Histogram_Bin" << i << "," << results.histogram[i] << "\n";
    }
    
    // Task 3: Sorting
    csv << "SortTime_ms," << results.sort_time_ms << "\n";
    
    // Task 4: Pearson Correlation
    csv << "PearsonCorrelation," << results.pearson_correlation << "\n";
    
    // Task 5: Moving Average
    csv << "MovingAverage_WindowSize," << results.window_size << "\n";
    csv << "MovingAverage_Count," << results.moving_average.size() << "\n";
    
    // Task 6: Outlier Detection
    csv << "Outlier_Threshold," << results.z_score_threshold << "\n";
    csv << "Outlier_Count," << results.outlier_count << "\n";
    if (results.outlier_count > 0 && results.outlier_count <= 10) {
        for (int i = 0; i < results.outlier_count; ++i) {
            csv << "Outlier_Value" << i << "," << results.outlier_values[i] << "\n";
        }
    }
    
    // Timing
    csv << "Task1_BasicStats_ms," << timings[0] << "\n";
    csv << "Task2_Histogram_ms," << timings[1] << "\n";
    csv << "Task3_Sorting_ms," << timings[2] << "\n";
    csv << "Task4_Correlation_ms," << timings[3] << "\n";
    csv << "Task5_MovingAverage_ms," << timings[4] << "\n";
    csv << "Task6_OutlierDetection_ms," << timings[5] << "\n";
    csv << "TotalTime_ms," << total_time << "\n";
    
    csv.close();
}

int main(int argc, char* argv[]) {
    // Parse command line
    long long datasetSize = 10000000;
    if (argc >= 2) {
        datasetSize = std::stoll(argv[1]);
    }
    
    std::cout << "=== SEQUENTIAL ANALYTICS (ALL 6 TASKS) ===" << std::endl;
    std::cout << "Dataset Size: " << datasetSize << " elements" << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    std::vector<double> timings(6, 0.0);
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Generate dataset
    std::cout << "Generating dataset..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> dataset = generateDataset(datasetSize);
    auto end = std::chrono::high_resolution_clock::now();
    double gen_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Generation time: " << gen_time << " ms" << std::endl;
    
    AnalyticsResults results;
    
    // TASK 1: Basic Statistics
    std::cout << "\n[Task 1] Computing Basic Statistics..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    computeBasicStats(dataset, results);
    end = std::chrono::high_resolution_clock::now();
    timings[0] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[0] << " ms" << std::endl;
    
    // TASK 2: Histogram
    std::cout << "[Task 2] Computing Histogram..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    computeHistogram(dataset, results, 20);
    end = std::chrono::high_resolution_clock::now();
    timings[1] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[1] << " ms" << std::endl;
    
    // TASK 3: Sorting
    std::cout << "[Task 3] Sorting data..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    computeSorting(dataset, results);
    end = std::chrono::high_resolution_clock::now();
    timings[2] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[2] << " ms" << std::endl;
    
    // TASK 4: Pearson Correlation
    std::cout << "[Task 4] Computing Pearson Correlation..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    results.pearson_correlation = computePearsonCorrelation(dataset);
    end = std::chrono::high_resolution_clock::now();
    timings[3] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[3] << " ms" << std::endl;
    
    // TASK 5: Moving Average
    std::cout << "[Task 5] Computing Moving Average..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    computeMovingAverage(dataset, results, 100);
    end = std::chrono::high_resolution_clock::now();
    timings[4] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[4] << " ms" << std::endl;
    
    // TASK 6: Outlier Detection
    std::cout << "[Task 6] Detecting Outliers..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    detectOutliers(dataset, results, 3.0);
    end = std::chrono::high_resolution_clock::now();
    timings[5] = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Time: " << timings[5] << " ms" << std::endl;
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // Print Results
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Task 1 - Basic Statistics:" << std::endl;
    std::cout << "  Count: " << results.count << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(4) << results.min_value << std::endl;
    std::cout << "  Max: " << results.max_value << std::endl;
    std::cout << "  Mean: " << results.mean << std::endl;
    std::cout << "  Median: " << results.median << std::endl;
    std::cout << "  StdDev: " << results.std_deviation << std::endl;
    
    std::cout << "\nTask 2 - Histogram:" << std::endl;
    std::cout << "  Bins: " << results.num_bins << std::endl;
    std::cout << "  Bin Width: " << results.bin_width << std::endl;
    
    std::cout << "\nTask 3 - Sorting:" << std::endl;
    std::cout << "  Sort Time: " << results.sort_time_ms << " ms" << std::endl;
    
    std::cout << "\nTask 4 - Pearson Correlation:" << std::endl;
    std::cout << "  Correlation: " << results.pearson_correlation << std::endl;
    
    std::cout << "\nTask 5 - Moving Average:" << std::endl;
    std::cout << "  Window Size: " << results.window_size << std::endl;
    std::cout << "  Output Points: " << results.moving_average.size() << std::endl;
    
    std::cout << "\nTask 6 - Outlier Detection:" << std::endl;
    std::cout << "  Z-Score Threshold: " << results.z_score_threshold << std::endl;
    std::cout << "  Outliers Found: " << results.outlier_count << std::endl;
    
    std::cout << "\n=== TIMING SUMMARY ===" << std::endl;
    std::cout << "Task 1 (Basic Stats):     " << timings[0] << " ms" << std::endl;
    std::cout << "Task 2 (Histogram):        " << timings[1] << " ms" << std::endl;
    std::cout << "Task 3 (Sorting):          " << timings[2] << " ms" << std::endl;
    std::cout << "Task 4 (Correlation):      " << timings[3] << " ms" << std::endl;
    std::cout << "Task 5 (Moving Average):   " << timings[4] << " ms" << std::endl;
    std::cout << "Task 6 (Outlier Detection):" << timings[5] << " ms" << std::endl;
    std::cout << "Total Time:                " << total_time << " ms" << std::endl;
    
    // Save to CSV
    std::string csv_filename = "sequential_results_" + std::to_string(datasetSize) + ".csv";
    saveToCSV(csv_filename, results, timings, total_time);
    std::cout << "\nResults saved to: " << csv_filename << std::endl;
    
    return 0;
}