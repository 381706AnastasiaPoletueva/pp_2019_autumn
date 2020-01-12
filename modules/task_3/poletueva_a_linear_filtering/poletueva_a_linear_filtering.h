// Copyright 2019 Poletueva Anastasia
#ifndef MODULES_TASK_3_POLETUEVA_A_LINEAR_FILTERING_POLETUEVA_A_LINEAR_FILTERING_H_
#define MODULES_TASK_3_POLETUEVA_A_LINEAR_FILTERING_POLETUEVA_A_LINEAR_FILTERING_H_

#define Error -1

#include<vector>

std::vector<int> LinFilterParallel(const std::vector <int> &a, int rows, int cols);
std::vector<int> LinFilter(const std::vector <int>& a, int rows, int cols);
std::vector<int> getRandomM(int rows, int cols);

#endif  // MODULES_TASK_3_POLETUEVA_A_LINEAR_FILTERING_POLETUEVA_A_LINEAR_FILTERING_H_
