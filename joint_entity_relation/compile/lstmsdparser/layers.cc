#include "lstmsdparser/layers.h"

BidirectionalLSTMLayer::BidirectionalLSTMLayer(dynet::Model& model,
	unsigned n_lstm_layers,
	unsigned dim_lstm_input,
	unsigned dim_hidden) :
	n_items(0),
	fw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
	bw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
	p_fw_guard(model.add_parameters({dim_lstm_input, 1})),
	p_bw_guard(model.add_parameters({dim_lstm_input, 1})){

	}

void BidirectionalLSTMLayer::new_graph(dynet::ComputationGraph* hg){
	fw_lstm.new_graph(*hg);
	bw_lstm.new_graph(*hg);
}

void BidirectionalLSTMLayer::add_inputs(dynet::ComputationGraph* hg,
	const std::vector<dynet::expr::Expression>& inputs){
	n_items = inputs.size();
	fw_lstm.start_new_sequence();
	bw_lstm.start_new_sequence();

	fw_lstm.add_input(dynet::expr::parameter(*hg, p_fw_guard));
	for (unsigned i = 0; i < n_items; ++i){
		fw_lstm.add_input(inputs[i]);
		bw_lstm.add_input(inputs[n_items - i - 1]);
	}
	bw_lstm.add_input(dynet::expr::parameter(*hg, p_bw_guard));
}

BidirectionalLSTMLayer::Output BidirectionalLSTMLayer::get_output(dynet::ComputationGraph *hg,
	int index){
	return std::make_pair(
		fw_lstm.get_h(dynet::RNNPointer(index + 1)).back(),
		bw_lstm.get_h(dynet::RNNPointer(n_items - index - 1)).back());
}

void BidirectionalLSTMLayer::get_outputs(dynet::ComputationGraph *hg,
	std::vector<BidirectionalLSTMLayer::Output>& outputs){
	outputs.resize(n_items);
	for (unsigned i = 0; i < n_items; ++i){
		outputs[i] = get_output(hg, i);
	}
}

void BidirectionalLSTMLayer::set_dropout(float& rate){
	fw_lstm.set_dropout(rate);
	bw_lstm.set_dropout(rate);
}

void BidirectionalLSTMLayer::disable_dropout(){
	fw_lstm.disable_dropout();
	bw_lstm.disable_dropout();
}