#ifndef __LSTMSDPARSER_SWAPBASED_H__
#define __LSTMSDPARSER_SWAPBASED_H__

#include "lstmsdparser/system.h"

struct SwapBased : public TransitionSystem{
public:
	explicit SwapBased();
	~SwapBased();
	
  void get_oracle_actions_calculate_orders(int root,
    const tree_t& tree,
    std::vector<int>& orders,
    int& timestamp);

  mpc_result_t get_oracle_actions_calculate_mpc(int root,
    const tree_t& tree,
    std::vector<int>& MPC);

  void get_swap_oracle_actions_onestep(graph_t& graph,
    const tree_t& tree,
    std::vector<int>& heads_rec,
    std::vector<int>& sigma,
    std::vector<int>& beta,
    std::vector<std::string>& actions,
    const std::vector<int>& orders,
    const std::vector<int>& MPC);

  void get_actions(graph_t& graph, std::vector<std::string>& gold_actions);
};


#endif  //  end for __LSTMSDPARSER_SWAPBASED_H__