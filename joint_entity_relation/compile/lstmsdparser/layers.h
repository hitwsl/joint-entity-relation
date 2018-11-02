#ifndef LAYERS_H_INCLUDE
#define LAYERS_H_INCLUDE

#include <vector>
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/lstm.h"
#include "dynet/expr.h"

struct BidirectionalLSTMLayer {
	typedef std::pair<dynet::expr::Expression, dynet::expr::Expression> Output;
	unsigned n_items;
	dynet::LSTMBuilder fw_lstm;
	dynet::LSTMBuilder bw_lstm;
	dynet::Parameter p_fw_guard;
	dynet::Parameter p_bw_guard;

	BidirectionalLSTMLayer() = default;

	explicit BidirectionalLSTMLayer(dynet::Model & model,
		unsigned n_lstm_layers,
		unsigned dim_lstm_input,
		unsigned dim_hidden);

	void new_graph(dynet::ComputationGraph* hg);
	void add_inputs(dynet::ComputationGraph* hg, 
		const std::vector<dynet::expr::Expression>& exprs);
	Output get_output(dynet::ComputationGraph* hg, int index);
	void get_outputs(dynet::ComputationGraph* hg, 
		std::vector<Output>& outputs);
	void set_dropout(float& rate);
	void disable_dropout();
};

#endif