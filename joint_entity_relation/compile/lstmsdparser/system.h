#ifndef __LTP_LSTMSDPARSER_SYSTEM_H__
#define __LTP_LSTMSDPARSER_SYSTEM_H__

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm> // for find

//! The tree type.
typedef std::vector<std::vector<int> > tree_t;
//! The MPC calculate result type
typedef std::tuple<bool, int, int> mpc_result_t;
typedef std::map<int, std::vector<std::pair<int, std::string>>> graph_t;

struct TransitionSystem{
  virtual void get_actions(graph_t& graph, std::vector<std::string>& gold_actions) = 0;
};


#endif  //  end for __LTP_LSTMSDPARSER_SYSTEM_H__