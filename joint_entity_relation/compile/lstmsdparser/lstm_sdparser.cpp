#include "lstmsdparser/lstm_sdparser.h"

namespace lstmsdparser {

    using namespace dynet::expr;
    using namespace dynet;
    using namespace std;
    namespace po = boost::program_options;

//struct LSTMParser {

    LSTMParser::LSTMParser() : Opt({255, 2, 100, 200, 50, 100, 200, 50, 50, 100, 10000, 25, 0.0,
                                    "sgd", "list-tree", "", "4000", "conll_ouput", true, false, false, false, false,
                                    false,
                                    true, false, false}) { }

    LSTMParser::~LSTMParser() { }

    void LSTMParser::set_options(Options opts) {
        this->Opt = opts;
    }

    bool LSTMParser::load(string model_file, string training_data_file, string word_embedding_file,
                          string dev_data_file) {
        this->transition_system = Opt.transition_system;
        corpus.set_transition_system(Opt.transition_system);
        if (DEBUG)
            cerr << "Loading training data from " << training_data_file << endl;
        //corpus.load_correct_actions(training_data_file);
        corpus.load_conll_file(training_data_file);
        if (corpus.maxChars > 255) {
            Opt.CHAR_SIZE = corpus.maxChars;
        }
        else {
            Opt.CHAR_SIZE = 255;
        }
        if (DEBUG)
            cerr << "finish loading training data" << endl;
        if (DEBUG)
            cerr << "Number of words: " << corpus.nwords << endl;
        kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
        if (word_embedding_file.length() > 0) {
            pretrained[kUNK] = std::vector<float>(Opt.PRETRAINED_DIM, 0);
            if (DEBUG)
                cerr << "Loading word embeddings from " << word_embedding_file << " with " << Opt.PRETRAINED_DIM <<
                " dimensions\n";
            ifstream in(word_embedding_file.c_str());
            if (!in)
                cerr << "### File does not exist! ###" << endl;
            string line;
            getline(in, line); // get the first info line
            std::vector<float> v(Opt.PRETRAINED_DIM, 0);
            string word;
            while (getline(in, line)) {
                istringstream lin(line);
                lin >> word;
                for (unsigned i = 0; i < Opt.PRETRAINED_DIM; ++i) lin >> v[i];
                unsigned id = corpus.get_or_add_word(word);
                pretrained[id] = v;
            }
        }

        get_dynamic_infos();
        if (DEBUG)
            cerr << "Setup model in dynet" << endl;
        //allocate memory for dynet
        char **dy_argv = new char *[6];
        int dy_argc = 3;
        dy_argv[0] = "dynet";
        dy_argv[1] = "--dynet-mem";
        dy_argv[2] = (char *) Opt.dynet_mem.c_str();
        if (Opt.dynet_seed.length() > 0) {
            dy_argc = 5;
            dy_argv[3] = "--dynet-seed";
            dy_argv[4] = (char *) Opt.dynet_seed.c_str();
        }
        dynet::initialize(dy_argc, dy_argv);
        delete dy_argv;
        ner_stack_lstm = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.HIDDEN_DIM, model),
        ner_out_lstm = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.HIDDEN_DIM, model),
        ent_lstm_fwd = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.LSTM_INPUT_DIM, model),
        ent_lstm_rev = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.LSTM_INPUT_DIM, model),
        fw_char_lstm = LSTMBuilder(Opt.LAYERS, Opt.LSTM_INPUT_DIM, 50, model), //Miguel
        bw_char_lstm = LSTMBuilder(Opt.LAYERS, Opt.LSTM_INPUT_DIM, 50, model),
        stack_lstm = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.HIDDEN_DIM, model);
        buffer_lstm = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.HIDDEN_DIM, model);
        pass_lstm = LSTMBuilder(Opt.LAYERS, 2 * Opt.BILSTM_HIDDEN_DIM, Opt.HIDDEN_DIM, model);
        action_lstm = LSTMBuilder(Opt.LAYERS, Opt.ACTION_DIM, Opt.HIDDEN_DIM, model);
        p_w = model.add_lookup_parameters(System_size.VOCAB_SIZE, {Opt.INPUT_DIM});
        p_a = model.add_lookup_parameters(System_size.ACTION_SIZE, {Opt.ACTION_DIM});
        p_r = model.add_lookup_parameters(System_size.ACTION_SIZE, {Opt.REL_DIM});
        p_pbias = model.add_parameters({Opt.HIDDEN_DIM});
        p_A = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_B = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_P = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_S = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_ner_S = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_cW = model.add_parameters({Opt.LSTM_INPUT_DIM * 2, Opt.LSTM_INPUT_DIM * 2});
        p_O = model.add_parameters({Opt.HIDDEN_DIM, Opt.HIDDEN_DIM});
        p_H = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, 2 * Opt.BILSTM_HIDDEN_DIM});
        p_H_t = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, 2 * Opt.BILSTM_HIDDEN_DIM});
        p_D = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, 2 * Opt.BILSTM_HIDDEN_DIM});
        p_D_t = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, 2 * Opt.BILSTM_HIDDEN_DIM});
        p_R = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, Opt.REL_DIM});
        p_R_t = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, Opt.REL_DIM});
        ner_p_R = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM, Opt.REL_DIM});
        p_w2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.INPUT_DIM});
        p_w2l_sp = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.INPUT_DIM});
        p_ib = model.add_parameters({Opt.LSTM_INPUT_DIM});
        p_cbias = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        p_cbias_t = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        ner_p_cbias = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        p_p2a = model.add_parameters({System_size.ACTION_SIZE, Opt.HIDDEN_DIM});
        p_action_start = model.add_parameters({Opt.ACTION_DIM});
        p_abias = model.add_parameters({System_size.ACTION_SIZE});
        p_buffer_guard = model.add_parameters({Opt.LSTM_INPUT_DIM});
        p_stack_guard = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        p_ner_stack_guard = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        p_ner_out_guard = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
        p_pass_guard = model.add_parameters({2 * Opt.BILSTM_HIDDEN_DIM});
//        if (Opt.USE_BILSTM) {
        buffer_bilstm = BidirectionalLSTMLayer(model, Opt.LAYERS, Opt.LSTM_INPUT_DIM, Opt.BILSTM_HIDDEN_DIM);
        //p_fwB = model.add_parameters({Opt.HIDDEN_DIM, Opt.BILSTM_HIDDEN_DIM});
        //p_bwB = model.add_parameters({Opt.HIDDEN_DIM, Opt.BILSTM_HIDDEN_DIM});
        p_biB = model.add_parameters({Opt.HIDDEN_DIM, 2 * Opt.BILSTM_HIDDEN_DIM});
        cerr << "Created Buffer BiLSTM" << endl;
//        }
        if (Opt.USE_TREELSTM) {
            tree_lstm = TheirTreeLSTMBuilder(1, Opt.LSTM_INPUT_DIM, Opt.LSTM_INPUT_DIM, model);
            cerr << "Created TreeLSTM" << endl;
        }
        if (Opt.USE_POS) {
            p_p = model.add_lookup_parameters(System_size.POS_SIZE, {Opt.POS_DIM});
            p_p2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.POS_DIM});
        }
        if (pretrained.size() > 0) {
            use_pretrained = true;
            p_t = model.add_lookup_parameters(System_size.VOCAB_SIZE, {Opt.PRETRAINED_DIM});
            for (auto it : pretrained)
                p_t.initialize(it.first, it.second);
            p_t2l = model.add_parameters({Opt.LSTM_INPUT_DIM, Opt.PRETRAINED_DIM});
        } else {
            use_pretrained = false;
            //p_t = nullptr;
            //p_t2l = nullptr;
        }
        if (Opt.USE_ATTENTION) {
            p_W_satb = model.add_parameters({1, Opt.HIDDEN_DIM + 2 * Opt.BILSTM_HIDDEN_DIM});
            p_bias_satb = model.add_parameters({1});
        }
        p_start_of_word = model.add_parameters({Opt.LSTM_INPUT_DIM}); //Miguel
        p_end_of_word = model.add_parameters({Opt.LSTM_INPUT_DIM}); //Miguel
        char_emb = model.add_lookup_parameters(Opt.CHAR_SIZE, {Opt.INPUT_DIM});//Miguel

        //this->model = model;
        if (model_file.length() > 0) {
            if (DEBUG)
                cerr << "loading model from " << model_file << endl;
            ifstream in(model_file.c_str());
            boost::archive::text_iarchive ia(in);
            ia >> this->model;
            if (DEBUG)
                cerr << "finish loading model" << endl;
        }
        if (dev_data_file.length() > 0) {
            if (DEBUG)
                cerr << "loading dev data from " << dev_data_file << endl;
            //corpus.load_correct_actionsDev(dev_data_file);
            corpus.load_conll_fileDev(dev_data_file);
            if (DEBUG)
                cerr << "finish loading dev data" << endl;
        }
        return true;
    }

    void LSTMParser::get_dynamic_infos() {

        System_size.kROOT_SYMBOL = corpus.get_or_add_word(lstmsdparser::ROOT_SYMBOL);

        {  // compute the singletons in the parser's training data
            map<unsigned, unsigned> counts;
            for (auto sent : corpus.sentences)
                for (auto word : sent.second) {
                    training_vocab.insert(word);
                    counts[word]++;
                }
            for (auto wc : counts)
                if (wc.second == 1) singletons.insert(wc.first);
        }
        if (DEBUG)
            cerr << "Number of words: " << corpus.nwords << endl;
        System_size.VOCAB_SIZE = corpus.nwords + 1;
        //ACTION_SIZE = corpus.nactions + 1;
        System_size.ACTION_SIZE = corpus.nactions + 30; // leave places for new actions in test set
        System_size.POS_SIZE =
                corpus.npos + 10;  // bad way of dealing with the fact that we may see new POS tags in the test set
        possible_actions.resize(corpus.nactions);
        for (unsigned i = 0; i < corpus.nactions; ++i)
            possible_actions[i] = i;
    }

    bool LSTMParser::has_path_to(int w1, int w2, const vector<vector<string>> &graph) {
        //cerr << endl << w1 << " has path to " << w2 << endl;
        if (graph[w1][w2] != REL_NULL)
            return true;
        for (int i = 0; i < (int) graph.size(); ++i) {
            if (graph[w1][i] != REL_NULL) if (has_path_to(i, w2, graph))
                return true;
        }
        return false;
    }

    bool LSTMParser::has_path_to_cycle(int w1, int w2, const vector<vector<string>> &graph) {
        //cerr << endl << w1 << " has path to " << w2 << endl;
        if (graph[w1][w2] != REL_NULL)
            return true;
        return false;
    }


    bool LSTMParser::has_path_to(int w1, int w2, const vector<vector<bool>> &graph) {
        if (graph[w1][w2])
            return true;
        for (int i = 0; i < (int) graph.size(); ++i) {
            if (graph[w1][i]) if (has_path_to(i, w2, graph))
                return true;
        }
        return false;
    }

    bool LSTMParser::has_path_to_cycle(int w1, int w2, const vector<vector<bool>> &graph) {
        //cerr << endl << w1 << " has path to " << w2 << endl;
        if (graph[w1][w2])
            return true;
        return false;
    }

    vector<unsigned> LSTMParser::get_children(unsigned id, const vector<vector<bool>> graph) {
        vector<unsigned> children;
        for (int i = 0; i < unsigned(graph[0].size()); i++) {
            if (graph[id][i])
                children.push_back(i);
        }
        return children;
    }

    bool LSTMParser::IsActionForbidden(bool is_ner, const string &a, unsigned bsize, unsigned ssize, unsigned ner_ssize,
                                       unsigned root,
                                       const vector<vector<bool>> dir_graph, //const vector<bool>  dir_graph [],
                                       const vector<int> &stacki, const vector<int> &bufferi,
                                       unsigned nr_root_rel) {
        int s0 = stacki.back();
        int b0 = bufferi.back();
//        int root_num = 0;
        int s0_head_num = 0;
        if (s0 >= 0)
            for (int i = 0; i < (int) dir_graph[root].size(); ++i)
                if (dir_graph[i][s0])
                    s0_head_num++;
        if (a[0] == 'L') {
            if (is_ner == false) {
                return true;
            }
            string rel = a.substr(3, a.size() - 4);
            if (bsize < 2 || ssize < 2) return true;

            if (Opt.IS_CYCLE) {
                if (has_path_to_cycle(s0, b0, dir_graph)) return true;
            }
            else {
                if (has_path_to(s0, b0, dir_graph)) return true;
            }
        }
        if (a[0] == 'R') {
            if (is_ner == false) {
                return true;
            }
            if (bsize < 2 || ssize < 2) return true;


            if (Opt.IS_CYCLE) {
                if (has_path_to_cycle(b0, s0, dir_graph)) return true;
            }
            else {
                if (has_path_to(b0, s0, dir_graph)) return true;
            }

//            if (b0 == (int)root) return true;
        }
        if (a[0] == 'N') {
            if (is_ner == false) {
                return true;
            }
            if (a[1] == 'S' && bsize < 2) return true;
            if (a[1] == 'R' && !(ssize > 1 && bsize > 1)) return true;
            if (a[1] == 'P' && !(ssize > 1 && bsize > 1)) return true;
        }
        if (a[0] == 'O') {
            if (is_ner == true) return true;
            if (ner_ssize > 1) return true;
            if (bsize == 1) return true;
        }
        if (a[0] == 'G') {
            if (a[1] == 'S') {
                if (is_ner == true) return true;
                if (bsize == 1) return true;
            }
            else if (a[1] == 'N') {
                if (is_ner == true) return true;
                if (ner_ssize == 1) return true;
            }
        }
        return false;
    }

    unsigned LSTMParser::UTF8Len(unsigned char x) {
        if (x < 0x80) return 1;
        else if ((x >> 5) == 0x06) return 2;
        else if ((x >> 4) == 0x0e) return 3;
        else if ((x >> 3) == 0x1e) return 4;
        else if ((x >> 2) == 0x3e) return 5;
        else if ((x >> 1) == 0x7e) return 6;
        else return 0;
    }


    void LSTMParser::compute_heads(const vector<unsigned> &sent, const vector<unsigned> &actions,
                                   std::vector<std::vector<string>> &graph,
                                   std::vector<std::pair<int, std::string>> &ref_ner) {
        const vector<string> &setOfActions = corpus.actions;
        unsigned sent_len = sent.size();
        // vector<vector<string>> graph;
        for (unsigned i = 0; i < sent_len; i++) {
            vector<string> r;
            for (unsigned j = 0; j < sent_len; j++) r.push_back(REL_NULL);
            graph.push_back(r);
        }
        vector<int> bufferi(sent_len + 1, 0), stacki(1, -999), passi(1, -999), ner_stacki(1, -999);
        for (unsigned i = 0; i < sent_len; ++i)
            bufferi[sent_len - i] = i;
        bufferi[0] = -999;
        bool is_ner = false;
        for (auto action: actions) { // loop over transitions for sentence
            const string &actionString = setOfActions[action];
            const char ac = actionString[0];
            const char ac2 = actionString[1];
            if (ac == 'N' && ac2 == 'S') {  // NO-SHIFT
                assert(bufferi.size() > 1 && is_ner); // dummy symbol means > 1 (not >= 1)
                int passi_size = (int) passi.size();
                for (int i = 1; i < passi_size; i++) {  //do not move pass_guard
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
                is_ner = false;
            } else if (ac == 'N' && ac2 == 'R') {
                assert(stacki.size() > 1 && is_ner);
                stacki.pop_back();
            } else if (ac == 'N' && ac2 == 'P') {
                assert(stacki.size() > 1 && is_ner);
                passi.push_back(stacki.back());
                stacki.pop_back();
            } else if (ac == 'L') { // LEFT-REDUCE or LEFT-PASS
                assert(stacki.size() > 1 && bufferi.size() > 1 && is_ner);
                unsigned depi, headi;
                depi = stacki.back();
                stacki.pop_back();
                headi = bufferi.back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                if (ac2 == 'P') { // LEFT-PASS
                    //TODO pass_lstm
                    passi.push_back(depi);
                }
            } else if (ac == 'R') { // RIGHT-SHIFT or RIGHT-PASSA
                assert(stacki.size() > 1 && bufferi.size() > 1 && is_ner);
                unsigned depi, headi;
                depi = bufferi.back();
                bufferi.pop_back();
                headi = stacki.back();
                stacki.pop_back();
                graph[headi][depi] = actionString.substr(3, actionString.size() - 4);
                if (ac2 == 'S') { //RIGHT-SHIFT
                    stacki.push_back(headi);
                    int passi_size = (int) passi.size();
                    for (int i = 1; i < passi_size; i++) {  //do not move pass_guard
                        stacki.push_back(passi.back());
                        passi.pop_back();
                    }
                    stacki.push_back(depi);
                    is_ner = false;
                }
                else if (ac2 == 'P') {
                    //TODO pass_lstm.add_input(nlcomposed);
                    passi.push_back(headi);
                    bufferi.push_back(depi);
                }
            } else if (ac == 'O') {
                assert(bufferi.size() > 1 && ner_stacki.size() == 1 && !is_ner);
                bufferi.pop_back();
            } else if (ac == 'G') {
                if (ac2 == 'S') {
                    assert(bufferi.size() > 1 && !is_ner);
                    ner_stacki.push_back(bufferi.back());
                    bufferi.pop_back();
                }
                else if (ac2 == 'N') {
                    assert(ner_stacki.size() > 1 && !is_ner);
                    bufferi.push_back(ner_stacki.back());
                    ref_ner[ner_stacki.back()] = std::make_pair(ner_stacki[1],
                                                                actionString.substr(3, actionString.size() - 4));
                    while (ner_stacki.size() > 1) {
                        ner_stacki.pop_back();
                    }
                    is_ner = true;
                }
            }
        }
        assert(bufferi.size() == 1);
    }

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
    vector<unsigned> LSTMParser::log_prob_parser(ComputationGraph *hg,
                                                 const vector<unsigned> &raw_sent,  // raw sentence
                                                 const vector<unsigned> &sent,  // sent with oovs replaced
                                                 const vector<string> &sent_str,  // sent with oovs replaced
                                                 const vector<unsigned> &sentPos,
                                                 const vector<unsigned> &correct_actions,
                                                 const vector<string> &setOfActions,
                                                 const map<unsigned, std::string> &intToWords,
                                                 double *right,
                                                 vector<vector<string>> &cand,
                                                 vector<Expression> *word_rep,
                                                 Expression *act_rep) {
        bool is_ner = false;
        vector<unsigned> results;
        const bool build_training_graph = correct_actions.size() > 0;
        bool apply_dropout = (Opt.DROPOUT && build_training_graph);
        vector<string> cv;
        for (unsigned i = 0; i < sent.size(); ++i)
            cv.push_back(REL_NULL);
        for (unsigned i = 0; i < sent.size(); ++i)
            cand.push_back(cv);
        stack_lstm.new_graph(*hg);
        ner_stack_lstm.new_graph(*hg);
        ner_out_lstm.new_graph(*hg);
        ent_lstm_fwd.new_graph(*hg);
        ent_lstm_rev.new_graph(*hg);
        action_lstm.new_graph(*hg);
        if (apply_dropout) {
            stack_lstm.set_dropout(Opt.DROPOUT);
            ner_stack_lstm.set_dropout(Opt.DROPOUT);
            ner_out_lstm.set_dropout(Opt.DROPOUT);
            ent_lstm_fwd.set_dropout(Opt.DROPOUT);
            ent_lstm_rev.set_dropout(Opt.DROPOUT);
            action_lstm.set_dropout(Opt.DROPOUT);
        } else {
            stack_lstm.disable_dropout();
            ner_stack_lstm.disable_dropout();
            ner_out_lstm.disable_dropout();
            ent_lstm_fwd.disable_dropout();
            ent_lstm_rev.disable_dropout();
            action_lstm.disable_dropout();
        }
        stack_lstm.start_new_sequence();
        ner_stack_lstm.start_new_sequence();
        ner_out_lstm.start_new_sequence();
        ent_lstm_fwd.start_new_sequence();
        ent_lstm_rev.start_new_sequence();
        action_lstm.start_new_sequence();
        pass_lstm.new_graph(*hg);
        if (apply_dropout) {
            pass_lstm.set_dropout(Opt.DROPOUT);
        }
        else {
            pass_lstm.disable_dropout();
        }
        pass_lstm.start_new_sequence();
        buffer_bilstm.new_graph(hg); // [bilstm] start_new_sequence is implemented in add_input
        if (apply_dropout) {
            buffer_bilstm.set_dropout(Opt.DROPOUT);
        }
        else {
            buffer_bilstm.disable_dropout();
        }
        Expression biB;
        biB = parameter(*hg, p_biB); // [bilstm]
        buffer_lstm.new_graph(*hg);
        if (apply_dropout) {
            buffer_lstm.set_dropout(Opt.DROPOUT);
        }
        else {
            buffer_lstm.disable_dropout();
        }
        buffer_lstm.start_new_sequence();
        Expression word_end = parameter(*hg, p_end_of_word); //Miguel
        Expression word_start = parameter(*hg, p_start_of_word); //Miguel

        if (Opt.USE_SPELLING) {
            fw_char_lstm.new_graph(*hg);
            bw_char_lstm.new_graph(*hg);
        }

        Expression W_satb; // [attention]
        Expression bias_satb;
        if (Opt.USE_ATTENTION) {
            W_satb = parameter(*hg, p_W_satb);
            bias_satb = parameter(*hg, p_bias_satb);
        }
        // variables in the computation graph representing the parameters
        Expression pbias = parameter(*hg, p_pbias);
        Expression H = parameter(*hg, p_H);
        Expression H_t = parameter(*hg, p_H_t);
        Expression D = parameter(*hg, p_D);
        Expression D_t = parameter(*hg, p_D_t);
        Expression R = parameter(*hg, p_R);
        Expression R_t = parameter(*hg, p_R_t);
        Expression ner_R = parameter(*hg, ner_p_R);
        Expression cbias = parameter(*hg, p_cbias);
        Expression cbias_t = parameter(*hg, p_cbias_t);
        Expression ner_cbias = parameter(*hg, ner_p_cbias);
        Expression cW = parameter(*hg, p_cW);
        Expression S = parameter(*hg, p_S);
        Expression ner_S = parameter(*hg, p_ner_S);
        Expression O = parameter(*hg, p_O);
        Expression B = parameter(*hg, p_B);
        Expression P = parameter(*hg, p_P);
        Expression A = parameter(*hg, p_A);
        Expression ib = parameter(*hg, p_ib);
        Expression w2l = parameter(*hg, p_w2l);
        Expression w2l_sp = parameter(*hg, p_w2l_sp);
        Expression p2l;
        if (Opt.USE_POS)
            p2l = parameter(*hg, p_p2l);
        Expression t2l;
        if (use_pretrained)
            t2l = parameter(*hg, p_t2l);
        Expression p2a = parameter(*hg, p_p2a);
        Expression abias = parameter(*hg, p_abias);
        Expression action_start = parameter(*hg, p_action_start);

        action_lstm.add_input(action_start);
        vector<Expression> buffer_origin(
                sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
        vector<Expression> buffer;  // variables representing word embeddings (possibly including POS info)
        vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
        // precompute buffer representation from left to right
        vector<Expression> word_emb(sent.size()); // [treelstm] store original word representation emb[i] for sent[i]
        for (unsigned i = 0; i < sent.size(); ++i) {
            assert(sent[i] < System_size.VOCAB_SIZE);
            string ww;
            if (build_training_graph) {
                ww = intToWords.at(raw_sent[i]);
            }
            else {
                ww = sent_str[i];
            }

            Expression w_sp;
            if (Opt.USE_SPELLING) {
                fw_char_lstm.start_new_sequence();
                fw_char_lstm.add_input(word_start);
                std::vector<int> strevbuffer;
                for (unsigned j = 0; j < ww.length(); j += UTF8Len(ww[j])) {
                    std::string wj;
                    for (unsigned h = j; h < j + UTF8Len(ww[j]); h++) wj += ww[h];
                    int wjint = corpus.charsToInt[wj];
                    strevbuffer.push_back(wjint);
                    Expression cj = lookup(*hg, char_emb, wjint);
                    fw_char_lstm.add_input(cj);
                }
                fw_char_lstm.add_input(word_end);
                Expression fw_i = fw_char_lstm.back();
                bw_char_lstm.start_new_sequence();
                bw_char_lstm.add_input(word_end);
                while (!strevbuffer.empty()) {
                    int wjint = strevbuffer.back();
                    Expression cj = lookup(*hg, char_emb, wjint);
                    bw_char_lstm.add_input(cj);
                    strevbuffer.pop_back();
                }
                bw_char_lstm.add_input(word_start);
                Expression bw_i = bw_char_lstm.back();
                vector<Expression> tt = {fw_i, bw_i};
                w_sp = concatenate(tt); //and this goes into the buffer...
            }
            Expression w = lookup(*hg, p_w, sent[i]);
            vector<Expression> args = {ib, w2l, w}; // learn embeddings
            if (Opt.USE_SPELLING) {
                args.push_back(w2l_sp);
                args.push_back(w_sp);
            }
            if (Opt.USE_POS) { // learn POS tag?
                Expression p = lookup(*hg, p_p, sentPos[i]);
                args.push_back(p2l);
                args.push_back(p);
            }
            if (use_pretrained) {  // include fixed pretrained vectors?
                if (pretrained.count(raw_sent[i])) {
                    Expression t = const_lookup(*hg, p_t, raw_sent[i]);
                    args.push_back(t2l);
                    args.push_back(t);
                }
                else {
                    unsigned lower_id = corpus.wordsToInt[cpyp::StrToLower(corpus.intToWords[raw_sent[i]])];
                    if (pretrained.count(lower_id)) {
                        Expression t = const_lookup(*hg, p_t, lower_id);
                        args.push_back(t2l);
                        args.push_back(t);
                        //cerr << "lower has embedding: " << corpus.intToWords[lower_id] << endl;
                    }
                }

            }
            buffer_origin[sent.size() - i] = rectify(affine_transform(args));
            bufferi[sent.size() - i] = i;
        }
        buffer_origin[0] = parameter(*hg, p_buffer_guard);
        bufferi[0] = -999;
        std::vector<BidirectionalLSTMLayer::Output> bilstm_outputs;
        buffer_bilstm.add_inputs(hg, buffer_origin);
        buffer_bilstm.get_outputs(hg,
                                  bilstm_outputs); // [bilstm] output of bilstm for buffer, first is fw, second is bw
        for (auto &b : bilstm_outputs)
            buffer.push_back(concatenate({b.first, b.second}));
        for (auto &b : buffer)
            buffer_lstm.add_input(b);
        vector<Expression> pass; //variables reperesenting embedding in pass buffer
        vector<int> passi; //position of words in pass buffer
        pass.push_back(parameter(*hg, p_pass_guard));
        passi.push_back(-999); // not used for anything
        pass_lstm.add_input(pass.back());
        vector<Expression> stack;  // variables representing subtree embeddings
        vector<int> stacki; // position of words in the sentence of head of subtree
        stack.push_back(parameter(*hg, p_stack_guard));
        stacki.push_back(-999); // not used for anything
        // drive dummy symbol on stack through LSTM
        stack_lstm.add_input(stack.back());
        vector<Expression> ner_stack;  // variables representing subtree embeddings
        vector<int> ner_stacki; // position of words in the sentence of head of subtree
        ner_stack.push_back(parameter(*hg, p_ner_stack_guard));
        ner_stacki.push_back(-999); // not used for anything
        // drive dummy symbol on stack through LSTM
        ner_stack_lstm.add_input(ner_stack.back());
        vector<Expression> ner_out;  // variables representing subtree embeddings
        vector<int> ner_outi; // position of words in the sentence of head of subtree
        ner_out.push_back(parameter(*hg, p_ner_out_guard));
        ner_outi.push_back(-999); // not used for anything
        // drive dummy symbol on stack through LSTM
        ner_out_lstm.add_input(ner_out.back());

        vector<Expression> log_probs;
//        string rootword;
        unsigned action_count = 0;  // incremented at each prediction
        //init graph connecting vector
        //vector<bool> dir_graph[sent.size()]; // store the connection between words in sent
        vector<vector<bool>> dir_graph;
        vector<bool> v;
        for (int i = 0; i < (int) sent.size(); i++) {
            v.push_back(false);
        }
        for (int i = 0; i < (int) sent.size(); i++) {
            dir_graph.push_back(v);
        }
        unsigned nr_root_rel = 0;
        while (bufferi.size() > 1 || ner_stacki.size() > 1) {
            // get list of possible actions for the current parser state
            vector<unsigned> current_valid_actions;
            for (auto a: possible_actions) {
                //cerr << " " << setOfActions[a]<< " ";
                if (IsActionForbidden(is_ner, setOfActions[a], buffer.size(), stack.size(), ner_stack.size(),
                                      sent.size() - 1,
                                      dir_graph, stacki, bufferi, nr_root_rel))
                    continue;
                //cerr << " <" << setOfActions[a] << "> ";
                current_valid_actions.push_back(a);
            }
            Expression p_t;
            /*
             * there are some big problem about the attention, it use token of stack to attend the tokens in the buffer, it is not very common sense.
             */
            if (Opt.USE_ATTENTION && Opt.USE_BILSTM) {
                vector<Expression> c(2);
                vector<Expression> alpha(bufferi.size() - 1);
                vector<Expression> buf_mat(bufferi.size() - 1);
                for (unsigned i = 1; i < bufferi.size(); ++i) { // buffer[0] is the guard
                    Expression b_fwh = bilstm_outputs[i].first;
                    Expression b_bwh = bilstm_outputs[i].second;
                    buf_mat[i - 1] = concatenate({b_fwh, b_bwh});
                    Expression s_b = concatenate({stack_lstm.back(), b_fwh, b_bwh});
                    alpha[i - 1] = rectify(affine_transform({bias_satb, W_satb, s_b}));
                }
                Expression a = concatenate(alpha);
                Expression ae = softmax(a);
                Expression buffer_mat = concatenate_cols(buf_mat);
                Expression buf_rep = buffer_mat * ae; // (2 * bilstm_hidden_dim , 1)
                p_t = affine_transform({pbias, S, stack_lstm.back(), ner_S, ner_stack_lstm.back(),
                                        O, ner_out_lstm.back(), P, pass_lstm.back(), biB, buf_rep, A,
                                        action_lstm.back()});
            }
            else if (Opt.USE_BILSTM) {
                Expression fwbuf, bwbuf;
                int idx;
                /*
                 * It need to be rechecked. Maybe there are somthing wrong about it.
                 */
                if (bufferi.size() > 1)
                    idx = sent.size() - bufferi.back();
                else
                    idx = 0;
                /*
                 * It need to be rechecked. It seems that it can directly use bilstm_outputs[0] when the bufferi size is 1
                 */
                fwbuf = bilstm_outputs[idx].first - bilstm_outputs[1].first;
                bwbuf = bilstm_outputs[1].second - bilstm_outputs[idx].second;
                // [bilstm] p_t = pbias + S * slstm + P * plstm + fwB * blstm_fw + bwB * blstm_bw + A * almst
                // [bilstm] p_t = pbias + S * slstm + P * plstm + biB * {blstm_fw + blstm_bw} + A * almst
                Expression bibuf = concatenate({fwbuf, bwbuf});
//                if (transition_system == "list-graph" || transition_system == "list-tree") {
                //p_t = affine_transform({pbias, S, stack_lstm.back(), P, pass_lstm.back(),
                //fwB, fwbuf, bwB, bwbuf, A, action_lstm.back()});
                p_t = affine_transform({pbias, S, stack_lstm.back(), ner_S, ner_stack_lstm.back(),
                                        O, ner_out_lstm.back(), P, pass_lstm.back(), biB, bibuf, A,
                                        action_lstm.back()});
            } else {
                /*
                 * it is very curious that why it does not use buffer_lstm when the condition is bi-lstm.
                 */
                p_t = affine_transform(
                        {pbias, S, stack_lstm.back(), ner_S, ner_stack_lstm.back(), O, ner_out_lstm.back(),
                         P, pass_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
            }
            Expression nlp_t = rectify(p_t);
            Expression r_t = affine_transform({abias, p2a, nlp_t});
            Expression adiste = log_softmax(r_t, current_valid_actions);
            vector<float> adist = as_vector(hg->incremental_forward(adiste));
            double best_score = adist[current_valid_actions[0]];
            unsigned best_a = current_valid_actions[0];
            for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
                if (adist[current_valid_actions[i]] > best_score) {
                    best_score = adist[current_valid_actions[i]];
                    best_a = current_valid_actions[i];
                }
            }
            unsigned action = best_a;
            if (build_training_graph) {  // if we have reference actions (for training) use the reference action
                action = correct_actions[action_count];
                if (best_a == action) { (*right)++; }
            }
            ++action_count;
            log_probs.push_back(pick(adiste, action));
            results.push_back(action);
            // add current action to action LSTM
            Expression actione = lookup(*hg, p_a, action);
            action_lstm.add_input(actione);
            // get relation embedding from action (TODO: convert to relation from action?)
            Expression relation = lookup(*hg, p_r, action);
            // do action
            const string &actionString = setOfActions[action];
            const char ac = actionString[0];
            const char ac2 = actionString[1];
            if (ac == 'N' && ac2 == 'S') {  // NO-SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                is_ner = false;
                int pass_size = (int) pass.size();
                for (int i = 1; i < pass_size; i++) {  //do not move pass_guard
                    stack.push_back(pass.back());
                    stack_lstm.add_input(pass.back());
                    pass.pop_back();
                    pass_lstm.rewind_one_step();
                    stacki.push_back(passi.back());
                    passi.pop_back();
                }
                stack.push_back(buffer.back());
                stack_lstm.add_input(buffer.back());
                buffer.pop_back();
                if (!Opt.USE_BILSTM)
                    buffer_lstm.rewind_one_step();
                stacki.push_back(bufferi.back());
                bufferi.pop_back();
            } else if (ac == 'N' && ac2 == 'R') {
                assert(stacki.size() > 1);
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
                // cerr << "label82" << endl;
            } else if (ac == 'N' && ac2 == 'P') {
                // cerr << "label91" << endl;
                assert(stacki.size() > 1);
                pass.push_back(stack.back());
                pass_lstm.add_input(stack.back());
                passi.push_back(stacki.back());
                stack.pop_back();
                stacki.pop_back();
                stack_lstm.rewind_one_step();
            } else if (ac == 'L') { // LEFT-REDUCE or LEFT-PASS
                // cerr << "labela1" << endl;
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = stack.back();
                depi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                head = buffer.back();
                headi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                Expression nlcomposed;
                /*
                 * this combinition only consider the head combinition, but ignorce the tail combination, which is not illgal and
                 * does not compare with the AAAI Paper.
                 * We need to correct it
                 */
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                nlcomposed = tanh(composed);
                stack_lstm.rewind_one_step();
                if (!Opt.USE_BILSTM) {
                    buffer_lstm.rewind_one_step();
                    buffer_lstm.add_input(nlcomposed);
                }
                buffer.push_back(nlcomposed);
                bufferi.push_back(headi);
                if (ac2 == 'P') { // LEFT-PASS
                    Expression nlcomposed_t;
                    Expression composed_t = affine_transform({cbias_t, D_t, dep, H_t, head, R_t, relation});
                    nlcomposed_t = tanh(composed_t);
                    pass_lstm.add_input(nlcomposed_t);
                    pass.push_back(nlcomposed_t);
                    passi.push_back(depi);
                }
            } else if (ac == 'R') { // RIGHT-SHIFT or RIGHT-PASSA
                // cerr << "labela3" << endl;
                assert(stacki.size() > 1 && bufferi.size() > 1);
                Expression dep, head;
                unsigned depi, headi;
                dep = buffer.back();
                depi = bufferi.back();
                buffer.pop_back();
                bufferi.pop_back();
                head = stack.back();
                headi = stacki.back();
                stack.pop_back();
                stacki.pop_back();
                dir_graph[headi][depi] = true; // add this arc to graph
                Expression nlcomposed;
                Expression nlcomposed_t;
                Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
                Expression composed_t = affine_transform({cbias_t, D_t, dep, H_t, head, R_t, relation});
                nlcomposed = tanh(composed);
                nlcomposed_t = tanh(composed_t);
                stack_lstm.rewind_one_step();
                if (!Opt.USE_BILSTM)
                    buffer_lstm.rewind_one_step();
                if (ac2 == 'S') { //RIGHT-SHIFT
                    is_ner = false;
                    stack_lstm.add_input(nlcomposed);
                    stack.push_back(nlcomposed);
                    stacki.push_back(headi);
                    int pass_size = (int) pass.size();
                    for (int i = 1; i < pass_size; i++) {  //do not move pass_guard
                        stack.push_back(pass.back());
                        stack_lstm.add_input(pass.back());
                        pass.pop_back();
                        pass_lstm.rewind_one_step();
                        stacki.push_back(passi.back());
                        passi.pop_back();
                    }
                    stack_lstm.add_input(nlcomposed_t);
                    stack.push_back(nlcomposed_t);
                    stacki.push_back(depi);
                }
                else if (ac2 == 'P') {
                    pass_lstm.add_input(nlcomposed);
                    pass.push_back(nlcomposed);
                    passi.push_back(headi);
                    if (!Opt.USE_BILSTM)
                        buffer_lstm.add_input(nlcomposed_t);
                    buffer.push_back(nlcomposed_t);
                    bufferi.push_back(depi);
                }
            }

            else if (ac == 'O') {
                // cerr << "labela5" << endl;
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                ner_outi.push_back(bufferi.back());
                ner_out.push_back(buffer.back());
                ner_out_lstm.add_input(buffer.back());
                buffer.pop_back();
                bufferi.pop_back();
                if (!Opt.USE_BILSTM) {
                    buffer_lstm.rewind_one_step();
                }
            }
            else if (ac == 'G') {
                if (ac2 == 'S') {
                    assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
                    ner_stack.push_back(buffer.back());
                    ner_stack_lstm.add_input(buffer.back());
                    ner_stacki.push_back(bufferi.back());
                    buffer.pop_back();
                    if (!Opt.USE_BILSTM) {
                        buffer_lstm.rewind_one_step();
                    }
                    bufferi.pop_back();
                }
                else if (ac2 == 'N') {
                    is_ner = true;
                    Expression previous;
                    Expression comp;
                    vector<Expression> entities(ner_stacki.size());
                    ent_lstm_fwd.start_new_sequence();
                    ent_lstm_rev.start_new_sequence();
                    for (unsigned i = 0; i < ner_stacki.size(); ++i) {
                        ent_lstm_fwd.add_input(ner_stack[i]);
                        ent_lstm_rev.add_input(ner_stack[ner_stacki.size() - i - 1]);
                    }
                    while (ner_stacki.size() > 1) {
                        ner_outi.push_back(ner_stacki.back());
                        ner_stack_lstm.rewind_one_step();
                        ner_stack.pop_back();
                        ner_stacki.pop_back();
                    }
                    Expression efwd = ent_lstm_fwd.back();
                    Expression erev = ent_lstm_rev.back();
                    if (apply_dropout) {
                        efwd = dropout(efwd, Opt.DROPOUT);
                        erev = dropout(erev, Opt.DROPOUT);
                    }
                    Expression c = concatenate({efwd, erev});
                    Expression composed = rectify(affine_transform({ner_cbias, cW, c, ner_R, relation}));
                    ner_out.push_back(composed);
                    /*
                     * there are big problem about ner_out_lstm. It seems no meaning.
                     */
                    ner_out_lstm.add_input(composed);
                    bufferi.push_back(ner_outi.back());
                    buffer.push_back(composed);
                    if (!Opt.USE_BILSTM) {
                        buffer_lstm.add_input(composed);
                    }
                }
            }
        }
        assert(buffer.size() == 1); // guard symbol
        assert(bufferi.size() == 1);
        Expression tot_neglogprob = -sum(log_probs);
        assert(tot_neglogprob.pg != nullptr);
        return results;
    }

    void LSTMParser::signal_callback_handler(int /* signum */) {
        if (requested_stop) {
            cerr << "\nReceived SIGINT again, quitting.\n";
            _exit(1);
        }
        cerr << "\nReceived SIGINT terminating optimization early...\n";
        requested_stop = true;
    }

    void LSTMParser::train(const std::string fname, const unsigned unk_strategy,
                           const double unk_prob) {
        requested_stop = false;
        signal(SIGINT, signal_callback_handler);
        unsigned status_every_i_iterations = 100;
        double best_LF = -1;
        bool softlinkCreated = false;
        Trainer *trainer;
        if (Opt.optimizer == "sgd") {
            trainer = new SimpleSGDTrainer(model);
            trainer->eta_decay = 0.08;
        }
        else if (Opt.optimizer == "adam") {
            trainer = new AdamTrainer(model);
        }
        //SimpleSGDTrainer sgd(model);
        //sgd.eta_decay = 0.08;
        std::vector<unsigned> order(corpus.nsentences);
        for (unsigned i = 0; i < corpus.nsentences; ++i)
            order[i] = i;
        double tot_seen = 0;
        status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
        unsigned si = corpus.nsentences;
        cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
        unsigned trs = 0;
        unsigned prd = 0; // number of predicted actions (in case of early update)
        double right = 0;
        double llh = 0;
        bool first = true;
        int iter = 0;
        //time_t time_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        time_t time_start = time(NULL);
        std::string t_s(asctime(localtime(&time_start)));
        cerr << "TRAINING STARTED AT: " << t_s.substr(0, t_s.size() - 1) << endl;
        while ((!requested_stop) && (iter < Opt.max_itr)) {
            ++iter;
            for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
                if (si == corpus.nsentences) {
                    si = 0;
                    if (first) { first = false; } else { trainer->update_epoch();/*sgd.update_epoch();*/ }
                    cerr << "**SHUFFLE\n";
                    random_shuffle(order.begin(), order.end());
                }
                tot_seen += 1;
                const std::vector<unsigned> &sentence = corpus.sentences[order[si]];
                std::vector<unsigned> tsentence = sentence;
                if (unk_strategy == 1) {
                    for (auto &w : tsentence)
                        if (singletons.count(w) && dynet::rand01() < unk_prob) w = kUNK;
                }
                const std::vector<unsigned> &sentencePos = corpus.sentencesPos[order[si]];
                const std::vector<unsigned> &actions = corpus.correct_act_sent[order[si]];
                ComputationGraph hg;
                //cerr << "Start word:" << corpus.intToWords[sentence[0]]<<corpus.intToWords[sentence[1]] << endl;
                std::vector<std::vector<string>> cand;
                log_prob_parser(&hg, sentence, tsentence, std::vector<string>(), sentencePos, actions, corpus.actions,
                                corpus.intToWords, &right, cand);
                double lp = as_scalar(hg.incremental_forward((VariableIndex) (hg.nodes.size() - 1)));
                if (lp < 0) {
                    cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
                    assert(lp >= 0.0);
                }
                hg.backward((VariableIndex) (hg.nodes.size() - 1));
                trainer->update(1.0);
                //sgd.update(1.0);
                llh += lp;
                ++si;
                trs += actions.size();
            }
            trainer->status();
            //sgd.status();
            //time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            time_t time_now = time(NULL);
            std::string t_n(asctime(localtime(&time_now)));
            cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentences)
            << " |time=" << t_n.substr(0, t_n.size() - 1) << ")\tllh: " << llh << " ppl: " << exp(llh / trs)
            << " err: " << (trs - right) / trs << endl;
            llh = trs = right = 0;

            static int logc = 0;
            ++logc;
            if (logc % Opt.evaluate_dim == 1) { // report on dev set
                unsigned dev_size = corpus.nsentencesDev;
                // dev_size = 100;
                double llh = 0;
                double trs = 0;
                double right = 0;
                //double correct_heads = 0;
                //double total_heads = 0;
                auto t_start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<unsigned>> gold_actions, pred_actions;
                std::vector<std::vector<std::vector<string>>> refs, hyps;
                std::vector<vector<std::pair<int, std::string>>> refs_ner, hyps_ner;
                for (unsigned sii = 0; sii < dev_size; ++sii) {
                    const std::vector<unsigned> &sentence = corpus.sentencesDev[sii];
                    const std::vector<string> &sentence_str = corpus.sentencesStrDev[sii];
                    const std::vector<unsigned> &sentencePos = corpus.sentencesPosDev[sii];
                    const std::vector<unsigned> &actions = corpus.correct_act_sentDev[sii];
                    std::vector<unsigned> tsentence = sentence;
                    for (auto &w : tsentence)
                        if (training_vocab.count(w) == 0) w = kUNK;
                    ComputationGraph hg;
                    std::vector<std::vector<string>> cand;
                    std::vector<unsigned> pred;
                    pred = log_prob_parser(&hg, sentence, tsentence, sentence_str, sentencePos, std::vector<unsigned>(),
                                           corpus.actions, corpus.intToWords, &right, cand);
                    double lp = 0;
                    llh -= lp;
                    trs += actions.size();
                    //cerr << "start word:" << sii << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]] << endl;
                    std::vector<std::vector<string>> ref;
                    std::vector<std::pair<int, std::string>> ref_ner(sentence.size(), std::make_pair(-1, "NULL"));
                    compute_heads(sentence, actions, ref, ref_ner);

                    std::vector<std::vector<string>> hyp;
                    std::vector<std::pair<int, std::string>> hyp_ner(sentence.size(), std::make_pair(-1, "NULL"));
                    compute_heads(sentence, pred, hyp, hyp_ner);
                    gold_actions.push_back(actions);
                    refs.push_back(ref);
                    refs_ner.push_back(ref_ner);
                    pred_actions.push_back(pred);
                    hyps.push_back(hyp);
                    hyps_ner.push_back(hyp_ner);
                }
                map<string, double> results = evaluate(refs, refs_ner, hyps, hyps_ner);
                auto t_end = std::chrono::high_resolution_clock::now();
                cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh
                << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " P-Entity:" <<
                results["P-Entity"] << " R-Entity:" << results["R-Entity"]
                << " F1-Entity: " << results["F1-Entity"] << " P-Head:" << results["P-Head"]
                << " R-Head:" << results["R-Head"] << " F1-Head:" << results["F1-Head"] << " P-relation:" <<
                results["P-relation"] << " R-relation:" << results["R-relation"] << " F1-relation:" <<
                results["F1-relation"]
                << "\t[" << dev_size << " sents in " <<
                std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]" << endl;

                if (results["F1-relation"] > best_LF) {
                    best_model_ss.str("");
                    boost::archive::text_oarchive oa(best_model_ss);
                    oa << model;
                    cerr << "best-result: " << results["F1-relation"] << endl;
                    best_LF = results["F1-relation"];
                    output_conll(refs, refs_ner, hyps, hyps_ner);
                    output_conll_actions(refs, refs_ner, gold_actions, hyps, hyps_ner, pred_actions);
                }
            }
        }//while
        std::ofstream out(fname);
        boost::archive::text_iarchive ia(best_model_ss);
        ia >> model;
        boost::archive::text_oarchive oa(out);
        oa << model;
        out.close();
    }


    void LSTMParser::predict_dev() {
        unsigned dev_size = corpus.nsentencesDev;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        //double correct_heads = 0;
        //double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<unsigned>> gold_actions, pred_actions;
        std::vector<std::vector<std::vector<string>>> refs, hyps;
        std::vector<vector<std::pair<int, std::string>>> refs_ner, hyps_ner;
        for (unsigned sii = 0; sii < dev_size; ++sii) {
            const std::vector<unsigned> &sentence = corpus.sentencesDev[sii];
            const std::vector<string> &sentence_str = corpus.sentencesStrDev[sii];
            const std::vector<unsigned> &sentencePos = corpus.sentencesPosDev[sii];
            const std::vector<unsigned> &actions = corpus.correct_act_sentDev[sii];
            std::vector<unsigned> tsentence = sentence;
            for (auto &w : tsentence)
                if (training_vocab.count(w) == 0) w = kUNK;
            ComputationGraph hg;
            std::vector<std::vector<string>> cand;
            std::vector<unsigned> pred;
            pred = log_prob_parser(&hg, sentence, tsentence, sentence_str, sentencePos, std::vector<unsigned>(),
                                   corpus.actions, corpus.intToWords, &right, cand);
            double lp = 0;
            llh -= lp;
            trs += actions.size();
            //cerr << "start word:" << sii << corpus.intToWords[sentence[0]] << corpus.intToWords[sentence[1]] << endl;
            std::vector<std::vector<string>> ref;
            std::vector<std::pair<int, std::string>> ref_ner(sentence.size(), std::make_pair(-1, "NULL"));
            compute_heads(sentence, actions, ref, ref_ner);

            std::vector<std::vector<string>> hyp;
            std::vector<std::pair<int, std::string>> hyp_ner(sentence.size(), std::make_pair(-1, "NULL"));
            compute_heads(sentence, pred, hyp, hyp_ner);
            gold_actions.push_back(actions);
            refs.push_back(ref);
            refs_ner.push_back(ref_ner);
            pred_actions.push_back(pred);
            hyps.push_back(hyp);
            hyps_ner.push_back(hyp_ner);
        }
        map<string, double> results = evaluate(refs, refs_ner, hyps, hyps_ner);
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << " P-Entity:" <<  results["P-Entity"] << " R-Entity:" << results["R-Entity"]
        << " F1-Entity: " << results["F1-Entity"] << " P-Head:" << results["P-Head"]
        << " R-Head:" << results["R-Head"] << " F1-Head:" << results["F1-Head"] << " P-relation:" <<
        results["P-relation"] << " R-relation:" << results["R-relation"] << " F1-relation:" <<
        results["F1-relation"] << endl;

    }

    map<string, double> LSTMParser::evaluate(const vector<vector<vector<string>>> &refs,
                                             const vector<vector<std::pair<int, std::string>>> &refs_ner,
                                             const vector<vector<vector<string>>> &hyps,
                                             const vector<vector<std::pair<int, std::string>>> &hyps_ner) {
        assert(refs.size() == hyps.size() && refs.size() == refs_ner.size() && hyps.size() == hyps_ner.size());
        int sum_correct_rels = 0; // labeled
        int sum_correct_head = 0; // labeled
        int sum_correct_entities = 0; // labeled
        int sum_gold_rels = 0;
        int sum_gold_head = 0;
        int sum_gold_entities = 0;
        int sum_pred_rels = 0;
        int sum_pred_head = 0;
        int sum_pred_entities = 0;

        for (int i = 0; i < (int) refs.size(); ++i) {
            unsigned sent_len = refs[i].size();
            for (unsigned j = 0; j < sent_len; ++j) {
                for (unsigned k = 0; k < sent_len; ++k) {
                    if (refs[i][j][k] != REL_NULL) {
                        sum_gold_head++;
                        sum_gold_rels++;
                        if (hyps[i][j][k] == refs[i][j][k]) {
                            sum_correct_head++;
                            if (refs_ner[i][j] == hyps_ner[i][j] && refs_ner[i][k] == hyps_ner[i][k]) {
                                sum_correct_rels++;
                            }
                        }
                    }
                    if (hyps[i][j][k] != REL_NULL) {
                        sum_pred_head++;
                        sum_pred_rels++;
                    }
                }//k
            }
        }//i

        for (int i = 0; i < (int) refs_ner.size(); ++i) {
            unsigned sent_len = refs_ner[i].size();
            for (unsigned j = 0; j < sent_len; ++j) {
                if (refs_ner[i][j].first > -1) {
                    sum_gold_entities++;
                    if (hyps_ner[i][j] == refs_ner[i][j]) {
                        sum_correct_entities++;
                    }
                }
                if (hyps_ner[i][j].first > -1) {
                    sum_pred_entities++;
                }
            }
        }//i
        map<string, double> result;
        result["P-Entity"] = sum_correct_entities * 100.0 / sum_pred_entities;
        result["R-Entity"] = sum_correct_entities * 100.0 / sum_gold_entities;
        result["F1-Entity"] = 2 * result["P-Entity"] * result["R-Entity"] / (result["P-Entity"] + result["R-Entity"]);

        result["P-Head"] = sum_correct_head * 100.0 / sum_pred_head;
        result["R-Head"] = sum_correct_head * 100.0 / sum_gold_head;
        result["F1-Head"] = 2 * result["P-Head"] * result["R-Head"] / (result["P-Head"] + result["R-Head"]);

        result["P-relation"] = sum_correct_rels * 100.0 / sum_pred_rels;
        result["R-relation"] = sum_correct_rels * 100.0 / sum_gold_rels;
        result["F1-relation"] =
                2 * result["P-relation"] * result["R-relation"] / (result["P-relation"] + result["R-relation"]);
        if (result["P-Entity"] == 0 && result["R-Entity"] == 0)
            result["F1-Entity"] = 0;
        if (result["P-Head"] == 0 && result["R-Head"] == 0)
            result["F1-Head"] = 0;
        if (result["P-relation"] == 0 && result["R-relation"] == 0)
            result["F1-relation"] = 0;

        return result;
    }


    void LSTMParser::output_conll(const vector<vector<vector<string>>> &refs,
                                  const vector<vector<std::pair<int, std::string>>> &refs_ner,
                                  const vector<vector<vector<string>>> &hyps,
                                  const vector<vector<std::pair<int, std::string>>> &hyps_ner) {
        assert(refs.size() == hyps.size() && refs.size() == refs_ner.size() && hyps.size() == hyps_ner.size());
        ofstream ofs(Opt.conll_result);
        for (int i = 0; i < (int) refs.size(); ++i) {
            const std::vector<string> &sentence_str = corpus.sentencesStrDev[i];
            const std::vector<unsigned> &sentencePos = corpus.sentencesPosDev[i];
            unsigned lenOfSentece = sentence_str.size();
            std::vector<string> gold_entity(lenOfSentece, "o");
            std::vector<string> pred_entity(lenOfSentece, "o");
            for (int j = 0; j < lenOfSentece; ++j) {
                if (refs_ner[i][j].first > -1) {
                    if (refs_ner[i][j].first == j) {
                        gold_entity[j] = "s-" + refs_ner[i][j].second;
                    }
                    else if ((j - refs_ner[i][j].first) == 1) {
                        gold_entity[refs_ner[i][j].first] = "b-" + refs_ner[i][j].second;
                        gold_entity[j] = "e-" + refs_ner[i][j].second;
                    }
                    else {
                        gold_entity[refs_ner[i][j].first] = "b-" + refs_ner[i][j].second;
                        for (int k = refs_ner[i][j].first + 1; k < j; k++) {
                            gold_entity[k] = "m" + refs_ner[i][j].second;
                        }
                        gold_entity[j] = "e-" + refs_ner[i][j].second;
                    }
                }
                if (hyps_ner[i][j].first > -1) {
                    if (hyps_ner[i][j].first == j) {
                        pred_entity[j] = "s-" + hyps_ner[i][j].second;
                    }
                    else if ((j - hyps_ner[i][j].first) == 1) {
                        pred_entity[hyps_ner[i][j].first] = "b-" + hyps_ner[i][j].second;
                        pred_entity[j] = "e-" + hyps_ner[i][j].second;
                    }
                    else {
                        pred_entity[hyps_ner[i][j].first] = "b-" + hyps_ner[i][j].second;
                        for (int k = hyps_ner[i][j].first + 1; k < j; k++) {
                            pred_entity[k] = "m" + hyps_ner[i][j].second;
                        }
                        pred_entity[j] = "e-" + hyps_ner[i][j].second;
                    }
                }
            }
            for (int j = 0; j < lenOfSentece; ++j) {
                ofs << sentence_str[j] << " " << sentencePos[j] << " " << gold_entity[j] <<
                " " << pred_entity[j] << endl;
            }

            for (int j = 0; j < lenOfSentece; ++j) {
                for (int k = 0; k < lenOfSentece; ++k) {
                    if (refs[i][j][k] != REL_NULL) {
                        if (j > k) {
                            ofs << "gold_rel " << k << " " << j << " -1 " << refs[i][j][k] << endl;
                        }
                        else {
                            ofs << "gold_rel " << j << " " << k << " 1 " << refs[i][j][k] << endl;
                        }
                    }
                    if (hyps[i][j][k] != REL_NULL) {
                        if (j > k) {
                            ofs << "pred_rel " << k << " " << j << " -1 " << hyps[i][j][k] << endl;
                        }
                        else {
                            ofs << "pred_rel " << j << " " << k << " 1 " << hyps[i][j][k] << endl;
                        }
                    }
                }//k
            }
            ofs << endl;
        }
        ofs.close();
        // return result;
    }
//    std::vector<std::vector<unsigned>> gold_actions, pred_actions;

    void LSTMParser::output_conll_actions(const vector<vector<vector<string>>> &refs,
                                          const vector<vector<std::pair<int, std::string>>> &refs_ner,
                                          const vector<vector<unsigned>> &gold_actions,
                                          const vector<vector<vector<string>>> &hyps,
                                          const vector<vector<std::pair<int, std::string>>> &hyps_ner,
                                          const vector<vector<unsigned>> &pred_actions) {
        assert(refs.size() == hyps.size() && refs.size() == refs_ner.size() && hyps.size() == hyps_ner.size());
        ofstream ofs(Opt.conll_result + "_actions");
        const vector<string> &setOfActions = corpus.actions;
        for (int i = 0; i < (int) refs.size(); ++i) {
            const std::vector<string> &sentence_str = corpus.sentencesStrDev[i];
            const std::vector<unsigned> &sentencePos = corpus.sentencesPosDev[i];
            unsigned lenOfSentece = sentence_str.size();
            std::vector<string> gold_entity(lenOfSentece, "o");
            std::vector<string> pred_entity(lenOfSentece, "o");
            bool pred_state = true;
            for (int j = 0; j < lenOfSentece; ++j) {
                if (refs_ner[i][j].first > -1) {
                    if (refs_ner[i][j] != hyps_ner[i][j]) {
                        pred_state = false;
                    }
                    if (refs_ner[i][j].first == j) {
                        gold_entity[j] = "s-" + refs_ner[i][j].second;
                    }
                    else if ((j - refs_ner[i][j].first) == 1) {
                        gold_entity[refs_ner[i][j].first] = "b-" + refs_ner[i][j].second;
                        gold_entity[j] = "e-" + refs_ner[i][j].second;
                    }
                    else {
                        gold_entity[refs_ner[i][j].first] = "b-" + refs_ner[i][j].second;
                        for (int k = refs_ner[i][j].first + 1; k < j; k++) {
                            gold_entity[k] = "m" + refs_ner[i][j].second;
                        }
                        gold_entity[j] = "e-" + refs_ner[i][j].second;
                    }
                }
                if (hyps_ner[i][j].first > -1) {
                    if (refs_ner[i][j] != hyps_ner[i][j]) {
                        pred_state = false;
                    }
                    if (hyps_ner[i][j].first == j) {
                        pred_entity[j] = "s-" + hyps_ner[i][j].second;
                    }
                    else if ((j - hyps_ner[i][j].first) == 1) {
                        pred_entity[hyps_ner[i][j].first] = "b-" + hyps_ner[i][j].second;
                        pred_entity[j] = "e-" + hyps_ner[i][j].second;
                    }
                    else {
                        pred_entity[hyps_ner[i][j].first] = "b-" + hyps_ner[i][j].second;
                        for (int k = hyps_ner[i][j].first + 1; k < j; k++) {
                            pred_entity[k] = "m" + hyps_ner[i][j].second;
                        }
                        pred_entity[j] = "e-" + hyps_ner[i][j].second;
                    }
                }
            }

            for (int j = 0; j < lenOfSentece; ++j) {
                for (int k = 0; k < lenOfSentece; ++k) {
                    if (refs[i][j][k] != hyps[i][j][k])
                        pred_state = false;
                }//k
            }

            if (pred_state == true)
                continue;
            for (int j = 0; j < lenOfSentece; ++j) {
                ofs << sentence_str[j] << " " << sentencePos[j] << " " << gold_entity[j] <<
                " " << pred_entity[j] << endl;
            }

            for (int j = 0; j < lenOfSentece; ++j) {
                for (int k = 0; k < lenOfSentece; ++k) {
                    if (refs[i][j][k] != REL_NULL) {
                        if (j > k) {
                            ofs << "gold_rel " << k << " " << j << " -1 " << refs[i][j][k] << endl;
                        }
                        else {
                            ofs << "gold_rel " << j << " " << k << " 1 " << refs[i][j][k] << endl;
                        }
                    }
                    if (hyps[i][j][k] != REL_NULL) {
                        if (j > k) {
                            ofs << "pred_rel " << k << " " << j << " -1 " << hyps[i][j][k] << endl;
                        }
                        else {
                            ofs << "pred_rel " << j << " " << k << " 1 " << hyps[i][j][k] << endl;
                        }
                    }
                }//k
            }
            ofs << "gold: ";
            for (auto action: gold_actions[i]) { // loop over transitions for sentence
                const string &actionString = setOfActions[action];
                ofs << actionString << " ";
            }
            ofs << endl;
            ofs << "pred: ";
            for (auto action: pred_actions[i]) { // loop over transitions for sentence
                const string &actionString = setOfActions[action];
                ofs << actionString << " ";
            }
            ofs << endl;
            ofs << endl;
        }
        ofs.close();
        // return result;
    }

} //  namespace lstmsdparser
