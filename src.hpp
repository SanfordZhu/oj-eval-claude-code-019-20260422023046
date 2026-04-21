#pragma once
#include "simulator.hpp"
#include <string>

namespace sjtu {

static Matrix *ConcatVertInPosition(const std::vector<Matrix *> &mats,
                                    GpuSimulator &gpu_sim,
                                    MatrixMemoryAllocator &alloc,
                                    Position pos,
                                    const std::string &base_name) {
  if (mats.empty()) return nullptr;
  Matrix *acc = alloc.Allocate(base_name + "_acc_0");
  gpu_sim.Copy(mats[0], acc, pos);
  for (size_t i = 1; i < mats.size(); ++i) {
    Matrix *next = alloc.Allocate(base_name + "_acc_" + std::to_string(i));
    gpu_sim.Concat(acc, mats[i], next, /*axis=*/0, pos);
    gpu_sim.ReleaseMatrix(acc);
    acc = next;
  }
  return acc;
}

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  const size_t total_rounds = keys.size();
  for (size_t i = 0; i < total_rounds; ++i) {
    Matrix *Q = rater.GetNextQuery();

    std::vector<Matrix *> k_slice(keys.begin(), keys.begin() + (i + 1));
    std::vector<Matrix *> v_slice(values.begin(), values.begin() + (i + 1));

    Matrix *K_cat_hbm = ConcatVertInPosition(k_slice, gpu_sim,
                                             matrix_memory_allocator,
                                             Position::kInGpuHbm,
                                             "Kcat_" + std::to_string(i));
    Matrix *V_cat_hbm = ConcatVertInPosition(v_slice, gpu_sim,
                                             matrix_memory_allocator,
                                             Position::kInGpuHbm,
                                             "Vcat_" + std::to_string(i));

    gpu_sim.MoveMatrixToSharedMem(Q);
    gpu_sim.MoveMatrixToSharedMem(K_cat_hbm);
    gpu_sim.MoveMatrixToSharedMem(V_cat_hbm);

    gpu_sim.Transpose(K_cat_hbm, Position::kInSharedMemory);

    Matrix *scores = matrix_memory_allocator.Allocate("scores_" + std::to_string(i));
    gpu_sim.MatMul(Q, K_cat_hbm, scores);

    Matrix *softmax_acc = nullptr;
    for (size_t r = 0; r < (i + 1); ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.GetRow(scores, r, row, Position::kInSharedMemory);

      Matrix *row_exp = matrix_memory_allocator.Allocate("rowexp_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatExp(row, row_exp);

      Matrix *row_sum = matrix_memory_allocator.Allocate("rowsum_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.Sum(row_exp, row_sum);

      Matrix *row_soft = matrix_memory_allocator.Allocate("rowsoft_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatDiv(row_exp, row_sum, row_soft);

      if (softmax_acc == nullptr) {
        softmax_acc = matrix_memory_allocator.Allocate("softmax_" + std::to_string(i) + "_acc");
        gpu_sim.Copy(row_soft, softmax_acc, Position::kInSharedMemory);
      } else {
        Matrix *softmax_new = matrix_memory_allocator.Allocate("softmax_" + std::to_string(i) + "_new_" + std::to_string(r));
        gpu_sim.Concat(softmax_acc, row_soft, softmax_new, /*axis=*/0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_acc);
        softmax_acc = softmax_new;
      }

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(row_soft);
    }

    Matrix *answer = matrix_memory_allocator.Allocate("answer_" + std::to_string(i));
    gpu_sim.MatMul(softmax_acc, V_cat_hbm, answer);

    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Free intermediates to reduce memory footprint
    gpu_sim.ReleaseMatrix(K_cat_hbm);
    gpu_sim.ReleaseMatrix(V_cat_hbm);
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(softmax_acc);

    gpu_sim.Run(false, &matrix_memory_allocator);

    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim, matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
