// Copyright 2019 Poletueva Anastasia
#include <mpi.h>
#include <ctime>
#include <algorithm>
#include <vector>
#include <random>
#include<iostream>
#include <utility>
#include "../../../modules/task_3/poletueva_a_linear_filtering/poletueva_a_linear_filtering.h"

std::vector<int> LinFilter(const std::vector <int>& a, int rows, int cols) {
  if (rows < 0 || cols < 0)
    throw Error;

  std::vector<int> res = a;
  std::vector<double> h = { 0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625 };

  int r = 1;
  for (int y = 1; y < cols - 1; y++)
    for (int x = 1; x < rows - 1; x++)
      for (int i = -r; i <= r; i++)
        for (int j = -r; j <= r; j++)
          res[x + y *rows] += a[x + i + (y + j)*rows] * h[i + 1 + (j + 1) * 3];
  return res;
}


std::vector<int> LinFilterParallel(const std::vector <int> &a, int rows, int cols) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rows < 0 || cols < 0)
    throw Error;

  if (size == 1) {
    return LinFilter(a, rows, cols);
  }

  std::vector<int> res = a;
  std::vector<int> ind(size);
  std::vector<std::pair<int, int> > vec{ { 1, 1 }, { 1, 2 }, { 1, 3 }, { 2, 2 },
  { 1, 5 }, { 2, 3 }, { 1, 7 }, { 2, 4 } };


  if (size == 2) {
    if (rank == 0) {
      res = LinFilter(a, rows, cols);
    }
    return res;
  }
  int delta_rows = vec[size - 1].second;
  int delta_cols = vec[size - 1].first;

  std::vector<int> localvec((rows / delta_rows + 1) * (cols / delta_cols + 1));

  if (rank == 0) {
    int procnum = 0;
    std::vector<int> sendvec;
    for (int r = 0; r < rows / delta_rows + 1; r++) {  // leftup
      int startind = r*cols;
      sendvec.insert(sendvec.end(), a.begin() + startind, a.begin() + startind + cols / delta_cols + 1);
    }
    ind[procnum] = cols + 1;
    procnum++;
    localvec = sendvec;
    sendvec.resize(0);
    for (int r = 0; r < rows / delta_rows + 1; r++) {  // rightup
      int startind = (delta_rows - 1) * rows / delta_rows + r*cols;
      sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols);
    }
    MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
    ind[procnum] = (delta_rows - 1) * rows / delta_rows + cols;
    procnum++;
    sendvec.resize(0);
    for (int r = -1; r < rows / delta_rows; r++) {  // leftdown
      int startind = (delta_cols - 1) * cols * rows / delta_cols + r*cols;
      sendvec.insert(sendvec.end(), a.begin() + startind, a.begin() + startind + cols / delta_cols + 1);
    }
    MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
    ind[procnum] = (delta_cols - 1) * cols * rows / delta_cols + 1;
    procnum++;
    sendvec.resize(0);
    for (int r = -1; r < rows / delta_rows; r++) {  // rightdown
      int startind = (delta_cols - 1) * cols * rows / delta_cols + (delta_rows - 1) * rows / delta_rows + r*cols;
      sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols);
    }
    MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
    ind[procnum] = (delta_cols - 1) * cols * rows / delta_cols + (delta_rows - 1) * rows / delta_rows;
    procnum++;
    for (int i = 1; i < delta_cols - 1; ++i) {  // left
      sendvec.resize(0);
      for (int r = -1; r < rows / delta_rows + 1; r++) {
        int startind = i * cols * rows / delta_cols + r*cols;
        sendvec.insert(sendvec.end(), a.begin() + startind, a.begin() + startind + cols / delta_cols + 1);
      }
      MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
      ind[procnum] = i * cols * rows / delta_cols + 1;
      procnum++;
    }
    for (int j = 1; j < delta_rows - 1; ++j) {  // up
      sendvec.resize(0);
      for (int r = 0; r < rows / delta_rows + 1; r++) {
        int startind = j * rows / delta_rows + r*cols;
        sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols + 1);
      }
      MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
      ind[procnum] = j * rows / delta_rows + cols;
      procnum++;
    }
    for (int i = 1; i < delta_cols - 1; ++i) {  // right
      sendvec.resize(0);
      for (int r = -1; r < rows / delta_rows + 1; r++) {
        int startind = i * cols * rows / delta_cols + delta_rows - 1 * rows / delta_rows + r*cols;
        sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols);
      }
      MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
      ind[procnum] = i * cols * rows / delta_cols + delta_rows - 1 * rows / delta_rows;
      procnum++;
    }
    for (int j = 1; j < delta_rows - 1; j++) {  // down
      sendvec.resize(0);
      for (int r = -1; r < rows / delta_rows; ++r) {
        int startind = delta_cols - 1 * cols * rows / delta_cols + j * rows / delta_rows + r*cols;
        sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols + 1);
      }
      MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
      ind[procnum] = delta_cols - 1 * cols * rows / delta_cols + j * rows / delta_rows;
      procnum++;
    }
    for (int i = 1; i < delta_cols - 1; ++i)
      for (int j = 1; j < delta_rows - 1; ++j) {
        sendvec.resize(0);
        for (int r = -1; r < rows / delta_rows + 1; ++r) {
          int startind = i * cols * rows / delta_cols + j * rows / delta_rows + r*cols;
          sendvec.insert(sendvec.end(), a.begin() + startind - 1, a.begin() + startind + cols / delta_cols + 1);
        }
        MPI_Send(&sendvec[0], (rows / delta_rows + 1) * (cols / delta_cols + 1), MPI_INT, procnum, 0, MPI_COMM_WORLD);
        ind[procnum] = i * cols * rows / delta_cols + j * rows / delta_rows;
        procnum++;
      }
  }

  if (rank != 0) {
    MPI_Status byaka;
    MPI_Recv(&localvec[0], localvec.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, &byaka);
  }

  // for (int i = 0; i < localvec.size(); ++i)
  // {
  // std::cout << rank  << " : " << localvec[i] << " ";
  // }
  // std::cout << std::endl;

  std::vector<int> local_res = LinFilter(localvec, (rows / delta_rows + 1), (cols / delta_cols + 1));

  // for (int i = 0; i < localvec.size(); ++i)
  // {
  // std::cout << rank << " : " << local_res[i] << " ";
  // }
  // std::cout << std::endl;

  if (rank != 0) {
    MPI_Send(&local_res[0], local_res.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  // for (int y = 1; y < rows / delta_rows; ++y)
  // for (int x = 1; x < cols / delta_cols; ++x)
  //  std::cout << rank << ": " <<local_res[x + y * (cols / delta_cols + 1)] << ' ';

  if (rank == 0) {
    // for (int i = 0; i < a.size(); ++i)
    // {
    // std::cout << a[i] << " ";
    // }
    // std::cout << std::endl;
    for (int y = 1; y < rows / delta_rows; ++y)
      for (int x = 1; x < cols / delta_cols; ++x)
        res[ind[0] + x - 1 + (y - 1) * cols] = local_res[x + y * (cols / delta_cols + 1)];
    // for (int y = 0; y < rows / delta_rows - 1; ++y)
    // for (int x = 0; x < cols / delta_cols - 1; ++x)
    //  std::cout << res[ind[0] + x + y * cols] << ' ';
    // std::cout << std::endl;
    // for (int y = 1; y < rows / delta_rows; ++y)
    // for (int x = 1; x < cols / delta_cols; ++x)
    //  std::cout << local_res[x + y * (cols / delta_cols + 1)] << "  ";
    // std::cout << std::endl;
    MPI_Status b;
    for (int i = 1; i < size; i++) {
      std::vector<int> lres(local_res.size());
      MPI_Recv(&lres[0], local_res.size(), MPI_INT, i, 0, MPI_COMM_WORLD, &b);
      // for (int y = 0; y < rows / delta_rows - 1; ++y)
      // for (int x = 0; x < cols / delta_cols - 1; ++x)
      //  std::cout << res[ind[i] + x + y * cols] << ' ';
      // std::cout << std::endl;
      // for (int y = 1; y < rows / delta_rows; ++y)
      // for (int x = 1; x < cols / delta_cols; ++x)
      //  std::cout << lres[x + y * (cols / delta_cols)] << "  ";
      // std::cout << std::endl;
      for (int y = 1; y < rows / delta_rows; ++y)
        for (int x = 1; x < cols / delta_cols; ++x)
          res[ind[i] + x - 1 + (y - 1) * cols] = lres[x + y * (cols / delta_cols + 1)];
    }
  }
  return res;
}

std::vector<int> getRandomM(int rows, int cols) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));

  std::vector<int>a(rows*cols);

  for (int i = 0; i < rows*cols; i++) {
    a[i] = gen() % 256;
  }
  return a;
}
