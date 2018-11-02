#ifndef TREELSTM_H_INCLUDE
#define TREELSTM_H_INCLUDE

#include "dynet/dynet.h"
#include "dynet/rnn.h"
#include "dynet/expr.h"

using namespace std;
using namespace dynet::expr;

namespace dynet {

class Model;

struct TheirTreeLSTMBuilder: public RNNBuilder {

    TheirTreeLSTMBuilder() = default;

    explicit TheirTreeLSTMBuilder(unsigned layers, unsigned input_dim,
            unsigned hidden_dim, Model& model);

    void set_dropout(float d) {
        dropout_rate = d;
    }

    // in general, you should disable dropout at test time
    void disable_dropout() {
        dropout_rate = 0;
    }

    Expression back() const override {
        return (cur == -1 ? h0.back() : h[cur].back());
    }

    vector<Expression> final_h() const override {
        return (h.size() == 0 ? h0 : h.back());
    }

    vector<Expression> final_s() const override {
        vector < Expression > ret = (c.size() == 0 ? c0 : c.back());
        for (auto my_h : final_h())
            ret.push_back(my_h);
        return ret;
    }

    unsigned num_h0_components() const override {
        return 2 * layers;
    }

    vector<Expression> get_h(RNNPointer i) const override {
        return (i == -1 ? h0 : h[i]);
    }

    vector<Expression> get_s(RNNPointer i) const override {
        vector < Expression > ret = (i == -1 ? c0 : c[i]);
        for (auto my_h : get_h(i))
            ret.push_back(my_h);
        return ret;
    }

    void copy(const RNNBuilder & params) override;

    Expression add_input(int idx, vector<unsigned> children,
            const Expression& x);

    void initialize_structure(unsigned sent_len);

protected:
    void new_graph_impl(ComputationGraph& cg) override;
    void start_new_sequence_impl(const vector<Expression>& h0) override;
    Expression add_input_impl(int idx, const Expression& x) override;
    Expression set_h_impl(int prev, const std::vector<Expression>& h_new) override{cerr<<"set_h_impl is not supported."<<endl;}
    Expression set_s_impl(int prev, const std::vector<Expression>& s_new) override{cerr<<"set_s_impl is not supported."<<endl;}

public:
    // first index is layer, then ...
    vector<vector<Parameter>> params;

    // first index is layer, then ...
    vector<vector<Expression>> param_vars;

    // first index is time, second is layer
    vector<vector<Expression>> h, c;

    // initial values of h and c at each layer
    // - both default to zero matrix input
    bool has_initial_state; // if this is false, treat h0 and c0 as 0
    vector<Expression> h0;
    vector<Expression> c0;
    unsigned layers;
    float dropout_rate;
};

} // namespace dynet

#endif