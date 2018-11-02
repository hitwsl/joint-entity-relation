#ifndef __LSTMSDPARSER_LISTBASED_H__
#define __LSTMSDPARSER_LISTBASED_H__

#include "lstmsdparser/system.h"

struct ListBased : public TransitionSystem{
public:
	graph_t graph;
	tree_t top_down_graph;
	std::vector<std::vector<bool>> subgraph;

	explicit ListBased();
	~ListBased();
	
	bool has_head(int w0, int w1);

	bool has_unfound_child(int w);

	//return if w has other head except the present one
	bool has_other_head(int w);

	//return if w has any unfound head
	bool lack_head(int w);

	//return if w has any unfound child in stack sigma 
	//!!! except the top in stack
	bool has_other_child_in_stack(std::vector<int>& sigma, int w);

	//return if w has any unfound head in stack sigma 
	//!!! except the top in stack
	bool has_other_head_in_stack(std::vector<int>& sigma, int w);

	//return the relation between child : w0, head : w1
	std::string get_arc_label(int w0, int w1);

	void get_list_oracle_actions_onestep(std::vector<int>& sigma,
    std::vector<int>& delta,
    std::vector<int>& beta,
    std::vector<std::string>& actions);

	void get_actions(graph_t& gold_graph,
                 std::vector<std::string>& gold_actions);

};

#endif  //  end for __LSTMSDPARSER_LISTBASED_H__