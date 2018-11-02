#include "lstmsdparser/swapbased.h"

SwapBased::SwapBased() {}

SwapBased::~SwapBased() {}

void SwapBased::get_oracle_actions_calculate_orders(int root,
    const tree_t& tree,
    std::vector<int>& orders,
    int& timestamp) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    orders[root] = timestamp;
    timestamp += 1;
    return;
  }

  int i;
  for (i = 0; i < children.size() && children[i] < root; ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }

  orders[root] = timestamp;
  timestamp += 1;

  for (; i < children.size(); ++ i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }
}

mpc_result_t SwapBased::get_oracle_actions_calculate_mpc(int root,
    const tree_t& tree,
    std::vector<int>& MPC) {
  const std::vector<int>& children = tree[root];
  if (children.size() == 0) {
    MPC[root] = root;
    return std::make_tuple(true, root, root);
  }

  int left = root, right = root;
  bool overall = true;

  int pivot = -1;
  for (pivot = 0; pivot < children.size() && children[pivot] < root; ++ pivot);

  for (int i = pivot - 1; i >= 0; -- i) {
    int child = children[i];
    mpc_result_t result =
      get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && std::get<2>(result) + 1 == left) {
      left = std::get<1>(result);
    } else {
      overall = false;
    }
  }

  for (int i = pivot; i < children.size(); ++ i) {
    int child = children[i];
    mpc_result_t result = get_oracle_actions_calculate_mpc(child, tree, MPC);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && right + 1 == std::get<1>(result)) {
      right = std::get<2>(result);
    } else {
      overall = false;
    }
  }

  for (int i = left; i <= right; ++ i) { MPC[i] = root; }

  return std::make_tuple(overall, left, right);
}

void SwapBased::get_swap_oracle_actions_onestep(graph_t& graph,
    const tree_t& tree,
    std::vector<int>& heads_rec,
    std::vector<int>& sigma,
    std::vector<int>& beta,
    std::vector<std::string>& actions,
    const std::vector<int>& orders,
    const std::vector<int>& MPC) {
  //! the head will be saved in heads record after been reduced
  
  if (sigma.size() < 2) {
    actions.push_back("SHIFT");
    sigma.push_back(beta.back()); beta.pop_back();
    return;
  }

  int top0 = sigma.back();
  int top1 = sigma[sigma.size() - 2];

  //INFO_LOG("step1 %d %d %d",top1,top0,beta.back());
  if (graph[top1].back().first == top0) {
    bool all_found = true;
    for (int c: tree[top1]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back("LEFT-ARC("+graph[top1].back().second+")");
      sigma.pop_back(); sigma.back() = top0; heads_rec[top1] = top0;
      return;
    }
  }
  if (graph[top0].back().first == top1) {
    bool all_found = true;
    for (int c: tree[top0]) { if (heads_rec[c] == -1) { all_found = false; } }
    if (all_found) {
      actions.push_back("RIGHT-ARC("+graph[top0].back().second+")");
      sigma.pop_back(); heads_rec[top0] = top1;
      return;
    }
  }
  int k = beta.empty() ? -1 : beta.back();
  if ((orders[top0] < orders[top1]) &&
      (k == -1 || MPC[top0] != MPC[k])) {
    actions.push_back("SWAP");
    sigma.pop_back(); sigma.back() = top0; beta.push_back(top1);
  } else {
    actions.push_back("SHIFT");
    sigma.push_back(beta.back()); beta.pop_back();
  }
}

void SwapBased::get_actions(graph_t& graph,
                 std::vector<std::string>& gold_actions){
  int N = graph.size();
  int root = -1;
  tree_t tree(N);
  for (int i = 0; i < N; ++ i) {
    int head = graph[i].back().first;
    if (head == -1) {
      if (root != -1)
        std::cerr << "error: there should be only one root." << std::endl;
      root = i;
    } else {
      tree[head].push_back(i);
    }
  }
  //! calculate the projective order
  int timestamp = 0;//!count for the order number
  std::vector<int> orders(N, -1);
  get_oracle_actions_calculate_orders(root,tree,orders,timestamp);
  std::vector<int> MPC(N, 0);
  get_oracle_actions_calculate_mpc(root,tree,MPC);
  gold_actions.clear();
  size_t len = N;
  std::vector<int> sigma;
  std::vector<int> beta;
  std::vector<int> heads_rec(N, -1);
  //std::vector<int> output(len, -1);

  //int step = 0;
  beta.push_back(-1);
  for (int i = N - 1; i >= 0; -- i) { beta.push_back(i); }
  while (!(sigma.size() ==1 && beta.empty())) {
    get_swap_oracle_actions_onestep(graph, tree, heads_rec, sigma, beta, gold_actions,orders,MPC);
  }
}
