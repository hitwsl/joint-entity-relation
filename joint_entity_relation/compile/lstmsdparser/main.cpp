//#include <iostream>

#include <cstring>
#include "lstmsdparser/lstm_sdparser.h"

using lstmsdparser::LSTMParser;
//namespace po = boost::program_options;

cpyp::Corpus corpus;

using namespace dynet::expr;
using namespace dynet;
using namespace std;
namespace po = boost::program_options;

//std::vector<unsigned> possible_actions;
//unordered_map<unsigned, std::vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
    ("dev_data,d", po::value<string>(), "Development corpus")
    ("conll_result", po::value<string>()->default_value("conll_result"), "Test corpus")
    ("test_data,p", po::value<string>(), "Test corpus")
    ("transition_system,s", po::value<string>()->default_value("list-graph"), "Transition system(swap - Arc-Standard, list-tree - List-Based tree, list-graph - List-Based graph)")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
    ("model,m", po::value<string>(), "Load saved model from this file")
    ("use_pos_tags,P", "make POS tags visible to parser")
    ("use_bilstm,B", "use bilstm for buffer")
    ("is_cycle", "does exit cycle")
    ("use_treelstm,R", "use treelstm for subtree in stack")
    ("use_attention,A", "use attention from stack top to whole buffer")
    ("data_type", po::value<string>()->default_value("sdpv2"), "Data type(sdpv2 - news, text - textbook), only for distinguishing model name")
    ("optimizer", po::value<string>()->default_value("sgd"), "Optimizer(sgd, adam)")
    ("dynet_seed", po::value<string>(), "Dynet seed for initialization, random initialization if not specified")
    ("dynet_mem", po::value<string>()->default_value("4000"), "Dynet memory size (MB) for initialization")
    ("model_dir", po::value<string>()->default_value(""), "Directory of model")
    ("max_itr", po::value<unsigned>()->default_value(10000), "Max training iteration")
    ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
     ("evaluate_dim", po::value<unsigned>()->default_value(25), "the numbers for evaluation")
    ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
    ("input_dim", po::value<unsigned>()->default_value(100), "input word embedding size (updated while training)")
    ("hidden_dim", po::value<unsigned>()->default_value(200), "lstm hidden dimension")
    ("bilstm_hidden_dim", po::value<unsigned>()->default_value(100), "bilstm hidden dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(100), "pretrained input word dimension")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
    ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
    //("ner_rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(200), "LSTM input dimension")
    ("dropout", po::value<float>(), "Dropout rate")
    ("use_spelling", "Use spelling model")
    ("post", "Should process headless words?")
    ("sdp_output", "Should output to sdp format? (CONLL08)")
    ("has_head", "Is every word required to have at least one head?")
    ("train,t", "Should training be run?")
    ("words,w", po::value<string>()->default_value(""), "Pretrained word embeddings")
    ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

int main(int argc, char** argv) {
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  
  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  lstmsdparser::Options Opt;
  Opt.optimizer = conf["optimizer"].as<string>();
  Opt.USE_POS = conf.count("use_pos_tags");
  Opt.USE_BILSTM = conf.count("use_bilstm");
  Opt.IS_CYCLE = conf.count("is_cycle");
  Opt.USE_SPELLING=conf.count("use_spelling");
  Opt.USE_TREELSTM = conf.count("use_treelstm");
  Opt.USE_ATTENTION = conf.count("use_attention");
  Opt.POST_PROCESS = conf.count("post");
  Opt.SDP_OUTPUT = conf.count("sdp_output");
  Opt.HAS_HEAD = conf.count("has_head");
  Opt.max_itr = conf["max_itr"].as<unsigned>();
  if (conf.count("dropout"))
    Opt.DROPOUT = conf["dropout"].as<float>();
  cerr << "Max training iteration: " << Opt.max_itr << endl;
  cerr << "Using " << Opt.optimizer << " as optimizer." << endl;
  if (Opt.USE_BILSTM)
    cerr << "Using bilstm for buffer." << endl;
  if (Opt.IS_CYCLE)
    cerr << "cycle exits for buffer." << endl;
  if (Opt.USE_SPELLING)
    cerr << "Using spelling for input." << endl;
  if (Opt.USE_TREELSTM)
    cerr << "Using treelstm for subtree in stack." << endl;
   if (Opt.USE_ATTENTION)
    cerr << "Using attention." << endl;
  if (Opt.POST_PROCESS)
    cerr << "Using post processing." << endl;
  if (Opt.HAS_HEAD)
    cerr << "Every word should have at least one head." << endl;
  Opt.transition_system = conf["transition_system"].as<string>();
  cerr << "Transition System: " << Opt.transition_system << endl;
  Opt.LAYERS = conf["layers"].as<unsigned>();
  Opt.INPUT_DIM = conf["input_dim"].as<unsigned>();
  Opt.PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  Opt.HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  Opt.evaluate_dim = conf["evaluate_dim"].as<unsigned>();
  Opt.ACTION_DIM = conf["action_dim"].as<unsigned>();
  Opt.LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  Opt.POS_DIM = conf["pos_dim"].as<unsigned>();
  Opt.REL_DIM = conf["rel_dim"].as<unsigned>();
  Opt.BILSTM_HIDDEN_DIM = conf["bilstm_hidden_dim"].as<unsigned>(); // [bilstm]
  Opt.conll_result = conf["conll_result"].as<string>();
  if (conf.count("dynet_seed")){
    Opt.dynet_seed = conf["dynet_seed"].as<string>();
  }
  if (conf.count("dynet_mem")){
    Opt.dynet_mem = conf["dynet_mem"].as<string>();
  }
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << conf["model_dir"].as<string>()
    << "parser_" << Opt.transition_system
    << '_' << Opt.optimizer
    << '_' << (Opt.USE_POS ? "pos" : "nopos")
    << '_' << (Opt.USE_BILSTM ? "bs" : "nobs")
    << '_' << (Opt.IS_CYCLE ? "cycle" : "nocycle")
    << '_' << (Opt.USE_SPELLING ? "spel" : "nospel")
    << '_' << (Opt.USE_TREELSTM ? "tr" : "notr")
    << '_' << (Opt.USE_ATTENTION ? "att" : "noatt")
    << '_' << conf["data_type"].as<string>()
    << '_' << Opt.LAYERS
    << '_' << Opt.INPUT_DIM
    << '_' << Opt.HIDDEN_DIM
    << '_' << Opt.ACTION_DIM
    << '_' << Opt.LSTM_INPUT_DIM
    << '_' << Opt.POS_DIM
    << '_' << Opt.REL_DIM
    << '_' << Opt.BILSTM_HIDDEN_DIM
    << "-pid" << getpid() << ".params";

  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

  LSTMParser *parser = new LSTMParser();
  //parser -> set_options(Opt);

  parser->DEBUG = true;
  if (conf.count("model") && conf.count("dev_data")){//for test
    parser -> set_options(Opt); // only for test
    parser -> load(conf["model"].as<string>(), conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }
  else{//for train
    parser -> set_options(Opt);
    parser -> load("", conf["training_data"].as<string>(), 
                  conf["words"].as<string>(), conf["dev_data"].as<string>() );
  }

  // OOV words will be replaced by UNK tokens
  //TRAINING
  
  if (conf.count("train")) {
    parser->train(fname, unk_strategy, unk_prob);
  } // should do training?
//
  if (conf.count("dev_data")) { // do test evaluation
    parser->predict_dev();
  }
//  if (conf.count("test_data")){
//    parser->test(conf["test_data"].as<string>());
//  }
}

