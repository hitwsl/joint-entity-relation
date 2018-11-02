#include "lstmsdparser/listbased.h"

ListBased::ListBased() {}

ListBased::~ListBased() {}

// return if w1 is one head of w0
bool ListBased::has_head(int w0, int w1){
  if (w0 <= 0) return false;
  for (auto w : graph[w0 - 1]){
    if (w.first == w1 - 1)
      return true;
  }
  return false;
}

bool ListBased::has_unfound_child(int w){
  //std::cerr << std::endl << "has unfound child: " << w << " ";
  for (auto child : top_down_graph[w]){
    //std::cerr << child << " , ";
    if (!subgraph[child][w])
      return true;
  }
  return false;
}

//return if w has other head except the present one
bool ListBased::has_other_head(int w){
  int head_num = 0;
  for (auto h : subgraph[w]){
    if (h) ++head_num;
  }
  //std::cerr << "has other head: " << w << " sub: " << head_num << " gold: " << graph[w].size() << std::endl;
  if (head_num + 1 < graph[w - 1].size())
    return true;
  return false;
}

//return if w has any unfound head
bool ListBased::lack_head(int w){
  if (w <= 0) return false;
  int head_num = 0;
  for (auto h : subgraph[w]){
    if (h) ++head_num;
  }
  if (head_num < graph[w - 1].size())
    return true;
  return false;
}

//return if w has any unfound child in stack sigma 
//!!! except the top in stack
bool ListBased::has_other_child_in_stack(std::vector<int>& sigma, int w){
  if (w <= 0) return false; // w = 0 is the root
  for (auto c : top_down_graph[w]){
    if (find(sigma.begin(), sigma.end(), c) != sigma.end() 
        && c!= sigma.back() && !subgraph[c][w])
      return true;
  }
  return false;
}

//return if w has any unfound head in stack sigma 
//!!! except the top in stack
bool ListBased::has_other_head_in_stack(std::vector<int>& sigma, int w){
  if (w <= 0) return false; // w = -1 is the root
  for (auto h : graph[w - 1]) {
    if (find(sigma.begin(), sigma.end(), h.first + 1) != sigma.end() 
        && (h.first + 1)!= sigma.back() && !subgraph[w][h.first + 1])
      return true;
  }
  return false;
}

//return the relation between child : w0, head : w1
std::string ListBased::get_arc_label(int w0, int w1){
  for (auto h : graph[w0 - 1]){
    if (h.first == w1 - 1){
      return h.second;
    }
  }
  std::cerr << "ERORR in list get_arc_label!" << std::endl;
  return "-ERORR-";
}

void ListBased::get_list_oracle_actions_onestep(std::vector<int>& sigma,
    std::vector<int>& delta,
    std::vector<int>& beta,
    std::vector<std::string>& actions) {
  int s0 = sigma.empty() ? -1 : sigma.back();
  int b0 = beta.empty() ? -1 : beta.back();
  //std::cerr << "s0: " << s0 << " b0: "<< b0 << std::endl;

  if(has_head(b0, b0)){
    if((get_arc_label(b0, b0) == "GS") || (get_arc_label(b0, b0) == "O-DELETE")){
      actions.push_back(get_arc_label(b0, b0));
      beta.pop_back();
      subgraph[b0][b0] = true;
      return;
    }
    else if(subgraph[b0][b0] == false){
      actions.push_back("GS");
      actions.push_back(get_arc_label(b0, b0));
      subgraph[b0][b0] = true;
      //return;
    }
  }

  if (s0 > 0 && has_head(s0, b0)) { // left s0 <- b0
    if ( !has_unfound_child(s0)
        && !has_other_head(s0)){
      actions.push_back("LR(" + get_arc_label(s0, b0) + ")");
      sigma.pop_back(); subgraph[s0][b0] = true;
      return;
    }
    else{ //has other child or head
      actions.push_back("LP(" + get_arc_label(s0, b0) + ")");
      delta.push_back(sigma.back()); sigma.pop_back(); subgraph[s0][b0] = true;
      return;
    }
  }
  else if ( s0 > 0 && has_head(b0, s0)) { //right arc s0 -> b0
    if ( !has_other_child_in_stack(sigma, b0)
        && !has_other_head_in_stack(sigma, b0)){
      actions.push_back("RS(" + get_arc_label(b0, s0) + ")");
      while (!delta.empty()){
        sigma.push_back(delta.back()); delta.pop_back();
      }
      sigma.push_back(beta.back()); beta.pop_back(); subgraph[b0][s0] = true;
      return;
    }
    else if (s0 > 0){
      actions.push_back("RP(" + get_arc_label(b0, s0) + ")");
      delta.push_back(sigma.back()); sigma.pop_back(); subgraph[b0][s0] = true;
      return;
    }
  }
  else if (!beta.empty()
          && !has_other_child_in_stack(sigma, b0)
          && !has_other_head_in_stack(sigma, b0) 
          ){
    actions.push_back("NS");
    while (!delta.empty()){
      sigma.push_back(delta.back()); delta.pop_back();
    }
    sigma.push_back(beta.back()); beta.pop_back();
    return;
  }
  else if ( s0 > 0
          && !has_unfound_child(s0)
          && !lack_head(s0)){
    actions.push_back("NR");
    sigma.pop_back();
    return;
  }
  else if ( s0 > 0){
    actions.push_back("NP");
    delta.push_back(sigma.back()); sigma.pop_back();
    return; 
  }
  else {
    actions.push_back("-E-");
    std::cerr << "error in oracle!" << std::endl;
    return;
  }
}

void ListBased::get_actions(graph_t& gold_graph,
                 std::vector<std::string>& gold_actions){
  // init graph, top_down_graph, subgraph
  int N = gold_graph.size();
  top_down_graph.clear();//std::vector<std::vector<int> > tree_t;
  top_down_graph.resize(N+1);
  graph.clear();//std::map<int, std::vector<std::pair<int, std::string>>> graph_t;
  graph = gold_graph;
  subgraph.clear();//std::vector<std::vector<bool>>
  std::vector<bool> v(N + 1, false);
  for (int i = 0; i < N + 1; ++i) subgraph.push_back(v);

  //int root = -1;
  // each id is +1 from graph, so when used in graph should -1
  for (int i = 0; i < N; ++ i) {
    for (auto n: graph[i]){
      int head = n.first + 1;
//      if (head == -1) {
//        if (root != -1)
//          std::cerr << "error: there should be only one root." << std::endl;
//        root = i + 1;
//      }
//      else {
        top_down_graph[head].push_back(i + 1);
//      }
    }
  }
  std::vector<int> sigma;
  std::vector<int> beta;
  std::vector<int> delta; // for pass action
  //beta.push_back(0);
  for (int i = N; i >= 1; -- i) { beta.push_back(i); }
  while (!beta.empty()) {
    get_list_oracle_actions_onestep(sigma, delta, beta, gold_actions);
      /*std::cerr << gold_actions.back() << std::endl;
      std::cerr << "stack: ";
      for (auto i : sigma) std::cerr << i << " , ";
      std::cerr << "pass: ";
      for (auto i : delta) std::cerr << i << " , ";
      std::cerr << "buffer: ";
      for (auto i : beta) std::cerr << i <<" , ";
      std::cerr << std::endl;*/
  }
}