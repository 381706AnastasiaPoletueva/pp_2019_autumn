// Copyright 2019 Poletueva Anastasia
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./poletueva_a_linear_filtering.h"

  TEST(Random_Filter_MPI, TEST1) {
  int rank;
  std::vector<int > global_res;
  std::vector<int> a;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
  a = getRandomM(4, 4);
  }
  global_res = LinFilterParallel(a, 4, 4);

  std::vector<int> reference_res;

  if (rank == 0) {
    reference_res = LinFilter(a, 4, 4);
    ASSERT_EQ(reference_res, global_res);
  }
  }

TEST(Contrast_Sequential_MPI, TEST2) {
  int rows = 4, cols = 4;
  std::vector <int> a2(rows * cols), res_a2(rows * cols);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    a2 = { 150, 118, 70, 120, 75, 200, 150, 200, 250, 162, 120, 65, 60, 100, 105, 80};
    res_a2 = { 150, 118, 70, 120, 75, 346, 288, 200, 250, 307, 244, 65, 60, 100, 105, 80};
  }
  std::vector <int> res(rows * cols);
  res = LinFilter(a2, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(res_a2, res);
  }
}

TEST(Contrast_Parallel_MPI, TEST3) {
  int rows = 4, cols = 4;
  std::vector <int> a3(rows * cols), res_a3(rows * cols);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    a3 = { 150, 118, 50, 120, 75, 200, 157, 200, 250, 152, 100, 65, 30, 120, 125, 90};
    res_a3 = { 150, 118, 50, 120, 75, 344, 291, 200, 250, 294, 222, 65, 30, 120, 125, 90};
  }
  std::vector <int> res(rows * cols);
  res = LinFilterParallel(a3, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(res_a3, res);
  }
}

TEST(Contrast_throw_MPI, TEST4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_ANY_THROW(getRandomM(-4, 2));
  }
}

TEST(Contrast_Parallel_throw_MPI, TEST5) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_ANY_THROW(getRandomM(2, -3));
  }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
