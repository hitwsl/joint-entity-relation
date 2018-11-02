#ifndef __LSTMSDPARSER_CPYPDICT_H__
#define __LSTMSDPARSER_CPYPDICT_H__

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <map>
#include <string>

#include "lstmsdparser/swapbased.h"
#include "lstmsdparser/listbased.h"

namespace cpyp {

  inline std::string StrToLower(const std::string s){
    std::string str = s;
    for (int i = 0; i < str.length(); ++i){
      str[i] = tolower(str[i]);
    }
    return str;
  }

class Corpus {
 //typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > Map;
// typedef std::unordered_map<unsigned,std::string, std::hash<std::string> > ReverseMap;
public: 
  bool DEBUG = false;
  bool USE_SPELLING=false;
  std::string transition_system; 

  std::map<int,std::vector<unsigned>> correct_act_sent;
  std::map<int,std::vector<unsigned>> sentences;
  std::map<int,std::vector<unsigned>> sentencesPos;

  std::map<int,std::vector<unsigned>> correct_act_sentDev;
  std::map<int,std::vector<unsigned>> sentencesDev;
  std::map<int,std::vector<unsigned>> sentencesPosDev;
  std::map<int,std::vector<std::string>> sentencesStrDev;
  std::map<int,std::vector<std::string>> sentencesCommDev; // comments starts with #
  unsigned nsentencesDev;

  std::map<int,std::vector<unsigned>> sentencesTest;
  std::map<int,std::vector<unsigned>> sentencesPosTest;
  std::map<int,std::vector<std::string>> sentencesStrTest;
  std::map<int,std::vector<std::string>> sentencesLemmaTest;
  std::map<int,std::vector<std::string>> sentencesXPosTest;
  std::map<int,std::vector<std::string>> sentencesFeatTest;
  std::map<int,std::vector<std::string>> sentencesMultTest;
 // std::map<int,std::vector<std::string>> sentencesCommTest;
  unsigned nsentencesTest;

  unsigned nsentences;
  unsigned nwords;
  unsigned nactions;
  unsigned npos;

  unsigned nsentencestest;
  unsigned nsentencesdev;
  int max;
  int maxPos;

  std::map<std::string, unsigned> wordsToInt;
  std::map<unsigned, std::string> intToWords;
  std::vector<std::string> actions;

  std::map<std::string, unsigned> posToInt;
  std::map<unsigned, std::string> intToPos;

  int maxChars;
  std::map<std::string, unsigned> charsToInt;
  std::map<unsigned, std::string> intToChars;

  // String literals
  static constexpr const char* UNK = "UNK";
  static constexpr const char* BAD0 = "<BAD0>";
  //static constexpr const char* ROOT = "ROOT";

  //! The tree type.
  typedef std::vector<std::vector<int> > tree_t;
  //! The MPC calculate result type
  typedef std::tuple<bool, int, int> mpc_result_t;

  Corpus() {
    max = 0;
    maxPos = 0;
    maxChars=0; //Miguel
  }


  inline unsigned UTF8Len(unsigned char x) {
    if (x < 0x80) return 1;
    else if ((x >> 5) == 0x06) return 2;
    else if ((x >> 4) == 0x0e) return 3;
    else if ((x >> 3) == 0x1e) return 4;
    else if ((x >> 2) == 0x3e) return 5;
    else if ((x >> 1) == 0x7e) return 6;
    else return 0;
  }


  void set_transition_system(std::string system){
    transition_system = system;
  }

  inline void split2(const std::string& source, const char& sep, 
      std::vector<std::string>& ret, int maxsplit=-1) {
    std::string str(source);
    int numsplit = 0;
    int len = str.size();
    size_t pos;
    for (pos = 0; pos < str.size() && (str[pos] == sep); ++ pos);
    str = str.substr(pos);

    ret.clear();
    while (str.size() > 0) {
      pos = std::string::npos;
      for (pos = 0; pos < str.size() && (str[pos] != sep); ++ pos);
      if (pos == str.size()) {
        pos = std::string::npos;
      }

      if (maxsplit >= 0 && numsplit < maxsplit) {
        ret.push_back(str.substr(0, pos));
        ++ numsplit;
      } else if (maxsplit >= 0 && numsplit == maxsplit) {
        ret.push_back(str);
        ++ numsplit;
      } else if (maxsplit == -1) {
        ret.push_back(str.substr(0, pos));
        ++ numsplit;
      }

      if (pos == std::string::npos) {
        str = "";
      } else {
        for (; pos < str.size() && (str[pos] == sep); ++ pos);
        str = str.substr(pos);
      }
    }
  }

  /**
  * Return a list of words of string str, the word are separated by
  * separator.
  *
  *  @param  str         std::string     the string
  *  @param  sep         char            the separator
  *  @param  maxsplit    std::string     the sep upperbound
  *  @return             std::vector<std::string> the words
  */
  inline std::vector<std::string> split2(const std::string& source, const char& sep, int maxsplit = -1) {
    std::vector<std::string> ret;
    split2(source, sep, ret, maxsplit);
    return ret;
  }


  inline void split(const std::string& source, std::vector<std::string>& ret,
                    int maxsplit=-1) {
    std::string str(source);
    int numsplit = 0;
    int len = str.size();
    size_t pos;
    for (pos = 0; pos < str.size() && (str[pos] == ' ' || str[pos] == '\t'
        || str[pos] == '\n' || str[pos] == '\r'); ++ pos);
    str = str.substr(pos);

    ret.clear();
    while (str.size() > 0) {
      pos = std::string::npos;
      for (pos = 0; pos < str.size() && (str[pos] != ' '
          && str[pos] != '\t'
          && str[pos] != '\r'
          && str[pos] != '\n'); ++ pos);
      if (pos == str.size()) pos = std::string::npos;
      if (maxsplit >= 0 && numsplit < maxsplit) {
        ret.push_back(str.substr(0, pos));
        ++ numsplit;
      } else if (maxsplit >= 0 && numsplit == maxsplit) {
        ret.push_back(str);
        ++ numsplit;
      } else if (maxsplit == -1) {
        ret.push_back(str.substr(0, pos));
        ++ numsplit;
      }

      if (pos == std::string::npos) {
        str = "";
      } else {
        for (; pos < str.size() && (str[pos] == ' '
            || str[pos] == '\t'
            || str[pos] == '\n'
            || str[pos] == '\r'); ++ pos);
        str = str.substr(pos);
      }
    }
  }

  /**
  * Return a list of words of string str, the word are separated by
  * separator.
  *
  *  @param  str         std::string     the string
  *  @param  maxsplit    std::string     the sep upperbound
  *  @return             std::vector<std::string> the words
  */
  inline std::vector<std::string> split(const std::string& source, int maxsplit = -1) {
    std::vector<std::string> ret;
    split(source, ret, maxsplit);
    return ret;
  }

  inline void load_conll_file(std::string file){
    std::ifstream actionsFile(file);
    //correct_act_sent=new vector<vector<unsigned>>();
    if (!actionsFile){
      std::cerr << "### File does not exist! ###" << std::endl;
    }
    std::string lineS;

    int sentence=0;
    wordsToInt[Corpus::BAD0] = 0;
    intToWords[0] = Corpus::BAD0;
    wordsToInt[Corpus::UNK] = 1; // unknown symbol
    intToWords[1] = Corpus::UNK;
    //wordsToInt[Corpus::ROOT] = 2; // root
    //intToWords[2] = Corpus::ROOT;
    //posToInt[Corpus::ROOT] = 1; // root
    //intToPos[1] = Corpus::ROOT;
    assert(max == 0);
    assert(maxPos == 0);
    max=2;
    maxPos=1;
  
    charsToInt[BAD0]=1;
    intToChars[1]="BAD0";
    maxChars=1;
  
    std::vector<unsigned> current_sent;
    std::vector<unsigned> current_sent_pos;
    bool is_tree = true;
    int ngraph = 0;
    std::map<int, std::vector<std::pair<int, std::string>>> graph;
    TransitionSystem* system;
//    if (transition_system == "swap")
//      system = new SwapBased();
    //else if (transition_system == "list-graph" || transition_system == "list-tree")
    system = new ListBased();
    while (getline(actionsFile, lineS)){
      ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
      ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
      if (lineS.empty()) {
        std::vector<std::string> gold_acts;
        system->get_actions(graph, gold_acts);
//        std::cerr << "actions start:" << std::endl;
//        for (auto g: gold_acts){
//          std::cerr << g << std::endl;
//        }
//        std::cerr << "actions end:" << std::endl;
        bool found=false;
        for (auto g: gold_acts){
          int i = 0;
          found=false;
          for (auto a: actions) {
            if (a==g) {
              std::vector<unsigned> a=correct_act_sent[sentence];
              a.push_back(i);
              correct_act_sent[sentence]=a;
              found=true;
              break;
            }
            ++i;
          }
          if (!found) {
            actions.push_back(g);
            std::vector<unsigned> a=correct_act_sent[sentence];
            a.push_back(actions.size()-1);
            correct_act_sent[sentence]=a;
          }
        }
        //current_sent.push_back(wordsToInt[Corpus::ROOT]);
        //current_sent_pos.push_back(posToInt[Corpus::ROOT]);
        sentences[sentence] = current_sent;
        sentencesPos[sentence] = current_sent_pos;    
        sentence++;
        nsentences = sentence;
      
        current_sent.clear();
        current_sent_pos.clear();
        graph.clear();
        if (!is_tree) ++ngraph;
        is_tree = true;
      } else {
        //stack and buffer, for now, leave it like this.
        // one line in each sentence may look like:
        // 5  American  american  ADJ JJ  Degree=Pos  6 amod  _ _
        // read the every line
        if (lineS[0] == '#') continue;
        std::vector<std::string> items = split2(lineS, '\t');
        if (items[0].find('-') != std::string::npos) continue;
        if (items[0].find('.') != std::string::npos) continue;
        unsigned id = std::atoi(items[0].c_str()) - 1;
        std::string word = items[1];
        //std::string word = StrToLower(items[1]);
        std::string pos = items[3];
        if (!(items[6] == "_" || items[7] == "_")){
          unsigned head = std::atoi(items[6].c_str()) - 1;
          std::string rel = items[7];
          graph[id].push_back(std::make_pair(head, rel));
        }
        if (graph[id].size() > 1) {is_tree = false; continue;}
        // new POS tag
        if (posToInt[pos] == 0) {
          posToInt[pos] = maxPos;
          intToPos[maxPos] = pos;
          npos = maxPos;
          maxPos++;
        }
        // new word
        if (wordsToInt[word] == 0) {
          wordsToInt[word] = max;
          intToWords[max] = word;
          nwords = max;
          max++;

          unsigned j = 0;
          while(j < word.length()) {
            std::string wj = "";
            for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
              wj += word[h];
            }
            if (charsToInt[wj] == 0) {
              charsToInt[wj] = maxChars;
              intToChars[maxChars] = wj;
              maxChars++;
            }
              j += UTF8Len(word[j]);
          }
        }
        current_sent.push_back(wordsToInt[word]);
        current_sent_pos.push_back(posToInt[pos]);
      }
    }

    // Add the last sentence.
    if (current_sent.size() > 0) {
      std::vector<std::string> gold_acts;
      system->get_actions(graph, gold_acts);
      bool found=false;
      for (auto g: gold_acts){
        int i = 0;
        found=false;
        for (auto a: actions) {
          if (a==g) {
            std::vector<unsigned> a=correct_act_sent[sentence];
            a.push_back(i);
            correct_act_sent[sentence]=a;
            found=true;
            break;
          }
          ++i;
        }
        if (!found) {
          actions.push_back(g);
          std::vector<unsigned> a=correct_act_sent[sentence];
          a.push_back(actions.size()-1);
          correct_act_sent[sentence]=a;
        }
      }
     // current_sent.push_back(wordsToInt[Corpus::ROOT]);
     // current_sent_pos.push_back(posToInt[Corpus::ROOT]);
      sentences[sentence] = current_sent;
      sentencesPos[sentence] = current_sent_pos;    
      sentence++;
      nsentences = sentence;
      if (!is_tree) ++ngraph;
    }
      
    actionsFile.close();
    std::cerr << "tree / total = " << nsentences - ngraph << " / " << nsentences << std::endl;
    if (DEBUG){
      std::cerr<<"done"<<"\n";
      for (auto a: actions) std::cerr<<a<<"\n";
    }
    nactions=actions.size();
    if (DEBUG){
      std::cerr<<"nactions:"<<nactions<<"\n";
      std::cerr<<"nwords:"<<nwords<<"\n";
      for (unsigned i=0;i<npos;i++) std::cerr<<i<<":"<<intToPos[i]<<"\n";
    }
    nactions=actions.size();
  }

  inline void load_conll_fileDev(std::string file){
    std::ifstream actionsFile(file);
    //correct_act_sent=new vector<vector<unsigned>>();
    if (!actionsFile){
      std::cerr << "### File does not exist! ###" << std::endl;
    }
    std::string lineS;

    assert(maxPos > 1);
    assert(max > 2);
    int sentence=0;
    std::vector<unsigned> current_sent;
    std::vector<unsigned> current_sent_pos;
    std::vector<std::string> current_sent_str;
   // std::vector<std::string> current_sent_comm;
    bool is_tree = true;
    int ngraph = 0;
    std::map<int, std::vector<std::pair<int, std::string>>> graph;
    TransitionSystem* system;
//    if (transition_system == "swap")
//      system = new SwapBased();
//    else if (transition_system == "list-graph" || transition_system == "list-tree")
    system = new ListBased();
    while (getline(actionsFile, lineS)){
      ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
      ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
      if (lineS.empty()) {
        /*for (unsigned i = 0; i < graph.size(); ++i){
          std::cerr << "word:" << i << std::endl;
          for (unsigned j = 0; j < graph[i].size(); ++j){
            std::cerr << graph[i][j].first << ":" << graph[i][j].second << std::endl;
          }
        }*/
        std::vector<std::string> gold_acts;
        system->get_actions(graph, gold_acts);
        bool found=false;
        for (auto g: gold_acts){
          auto actionIter = std::find(actions.begin(), actions.end(), g);
          if (actionIter != actions.end()) {
            unsigned actionIndex = std::distance(actions.begin(), actionIter);
            correct_act_sentDev[sentence].push_back(actionIndex);
          } else {
          // new action
            BOOST_ASSERT_MSG(false, "Unknow action in development data.");
//            actions.push_back(g);
//            unsigned actionIndex = actions.size() - 1;
//            correct_act_sentDev[sentence].push_back(actionIndex);
          }
        }
       // current_sent.push_back(wordsToInt[Corpus::ROOT]);
        //current_sent_pos.push_back(posToInt[Corpus::ROOT]);
        //current_sent_str.push_back("");
        //current_sent_str.push_back(Corpus::ROOT);
        sentencesDev[sentence] = current_sent;
        sentencesPosDev[sentence] = current_sent_pos;
        sentencesStrDev[sentence] = current_sent_str;
        //sentencesCommDev[sentence] = current_sent_comm;
        sentence++;
        nsentencesDev = sentence;

        current_sent.clear();
        current_sent_pos.clear();
        current_sent_str.clear();
        //current_sent_comm.clear();
        graph.clear();
        if (!is_tree) ++ngraph;
        is_tree = true;
      } else {
        //stack and buffer, for now, leave it like this.
        // one line in each sentence may look like:
        // 5  American  american  ADJ JJ  Degree=Pos  6 amod  _ _
        // read the every line
        if (lineS[0] == '#') {
         // current_sent_comm.push_back(lineS);
          continue;
        }
        std::vector<std::string> items = split2(lineS,'\t');
        if (items[0].find('-') != std::string::npos) continue;
        if (items[0].find('.') != std::string::npos) continue;
        unsigned id = std::atoi(items[0].c_str()) - 1;
        std::string word = items[1];
        //std::string word = StrToLower(items[1]);
        std::string pos = items[3];
        if (!(items[6] == "_" || items[7] == "_")){
          unsigned head = std::atoi(items[6].c_str()) - 1;
          std::string rel = items[7];
          graph[id].push_back(std::make_pair(head, rel));
        }
        if (graph[id].size() > 1) {is_tree = false; continue;}
        // new POS tag
        if (posToInt[pos] == 0) {
//          posToInt[pos] = maxPos;
//          intToPos[maxPos] = pos;
//          npos = maxPos;
//          maxPos++;
          BOOST_ASSERT_MSG(false, "Unknow postag in development data.");
        }
        // add an empty string for any token except OOVs (it is easy to 
        // recover the surface form of non-OOV using intToWords(id)).
        current_sent_str.push_back(word);
        // OOV word
        if (wordsToInt[word] == 0) {
//          if (USE_SPELLING) {
//            max = nwords + 1;
//            wordsToInt[word] = max;
//            intToWords[max] = word;
//            nwords = max;
//          } else {
//            // save the surface form of this OOV before overwriting it.
//            current_sent_str[current_sent_str.size()-1] = word;
            word = Corpus::UNK;
//          }
        }
        //current_sent_str[current_sent_str.size()-1] = items[1]; // save word for [lower emb]
        current_sent.push_back(wordsToInt[word]);
        current_sent_pos.push_back(posToInt[pos]);
      }
    }

    // Add the last sentence.
    if (current_sent.size() > 0) {
      std::vector<std::string> gold_acts;
      system->get_actions(graph, gold_acts);
      bool found=false;
      for (auto g: gold_acts){
        auto actionIter = std::find(actions.begin(), actions.end(), g);
        if (actionIter != actions.end()) {
          unsigned actionIndex = std::distance(actions.begin(), actionIter);
          correct_act_sentDev[sentence].push_back(actionIndex);
        } else {
          // new action
          BOOST_ASSERT_MSG(false, "Unknow action in development data.");
//          actions.push_back(g);
//          unsigned actionIndex = actions.size() - 1;
//          correct_act_sentDev[sentence].push_back(actionIndex);
        }
      }
      //current_sent.push_back(wordsToInt[Corpus::ROOT]);
      //current_sent_pos.push_back(posToInt[Corpus::ROOT]);
      //current_sent_str.push_back("");
      //current_sent_str.push_back(Corpus::ROOT);
      sentencesDev[sentence] = current_sent;
      sentencesPosDev[sentence] = current_sent_pos;
      sentencesStrDev[sentence] = current_sent_str;
     // sentencesCommDev[sentence] = current_sent_comm;
      sentence++;
      nsentencesDev = sentence;
      if (!is_tree) ++ngraph;
    }
      
    actionsFile.close();
    std::cerr << "tree / total = " << nsentencesDev - ngraph << " / " << nsentencesDev << std::endl;
    if (DEBUG){
      std::cerr<<"done"<<"\n";
      for (auto a: actions) std::cerr<<a<<"\n";
    }
    nactions=actions.size();
    if (DEBUG){
      std::cerr<<"nactions:"<<nactions<<"\n";
      std::cerr<<"nwords:"<<nwords<<"\n";
      for (unsigned i=0;i<npos;i++) std::cerr<<i<<":"<<intToPos[i]<<"\n";
    }
    nactions=actions.size();
  }

//  inline void load_conll_fileTest(std::string file){
//    std::ifstream actionsFile(file);
//    if (!actionsFile){
//      std::cerr << "### File does not exist! ###" << std::endl;
//    }
//    std::string lineS;
//
//    assert(maxPos > 1);
//    assert(max > 3);
//    int sentence=0;
//    std::vector<unsigned> current_sent;
//    std::vector<unsigned> current_sent_pos;
//    std::vector<std::string> current_sent_str;
//    std::vector<std::string> current_sent_lemma;
//    std::vector<std::string> current_sent_xpos;
//    std::vector<std::string> current_sent_feat;
//    std::vector<std::string> current_sent_mult; // for multiword like "1-2"
//    //std::vector<std::string> current_sent_comm;
//    while (getline(actionsFile, lineS)){
//      ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
//      ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
//      if (lineS.empty()) {
//        //current_sent.push_back(wordsToInt[Corpus::ROOT]);
//        //current_sent_pos.push_back(posToInt[Corpus::ROOT]);
//        //current_sent_str.push_back("");
//        sentencesTest[sentence] = current_sent;
//        sentencesPosTest[sentence] = current_sent_pos;
//        sentencesStrTest[sentence] = current_sent_str;
//        sentencesLemmaTest[sentence] = current_sent_lemma;
//        sentencesXPosTest[sentence] = current_sent_xpos;
//        sentencesFeatTest[sentence] = current_sent_feat;
//        sentencesMultTest[sentence] = current_sent_mult;
//       // sentencesCommTest[sentence] = current_sent_comm;
//        sentence++;
//        nsentencesTest = sentence;
//
//        current_sent.clear();
//        current_sent_pos.clear();
//        current_sent_str.clear();
//        current_sent_lemma.clear();
//        current_sent_xpos.clear();
//        current_sent_feat.clear();
//        current_sent_mult.clear();
//        //current_sent_comm.clear();
//      } else {
//      //stack and buffer, for now, leave it like this.
//      // one line in each sentence may look like:
//      // 5  American  american  ADJ JJ  Degree=Pos  6 amod  _ _
//      // read the every line
//        if (lineS[0] == '#') {
//         // current_sent_comm.push_back(lineS);
//          continue;
//        }
//        std::vector<std::string> items = split2(lineS,'\t');
//        if (items[0].find('.') != std::string::npos) continue;
//        current_sent_mult.push_back("");
//        std::size_t found_mult = items[0].find('-');
//        if (found_mult!=std::string::npos){
//          unsigned first_id = std::atoi(items[0].substr(0, found_mult).c_str()) - 1;
//          current_sent_mult[first_id] = lineS;
//          continue;
//        }
//        unsigned id = std::atoi(items[0].c_str()) - 1;
//        //std::string word = StrToLower(items[1]);
//        std::string word = items[1];
//        std::string pos = items[3];
//        if ( id < current_sent.size()) continue; // in case that the input has multihead
//        current_sent_lemma.push_back(items[2]);
//        current_sent_xpos.push_back(items[4]);
//        current_sent_feat.push_back(items[5]);
//        // new POS tag
//        if (posToInt[pos] == 0) {
//          posToInt[pos] = maxPos;
//          intToPos[maxPos] = pos;
//          npos = maxPos;
//          maxPos++;
//        }
//        // add an empty string for any token except OOVs (it is easy to
//        // recover the surface form of non-OOV using intToWords(id)).
//        current_sent_str.push_back("");
//        // OOV word
//        if (wordsToInt[word] == 0) {
//          if (USE_SPELLING) {
//            max = nwords + 1;
//            wordsToInt[word] = max;
//            intToWords[max] = word;
//            nwords = max;
//          } else {
//            // save the surface form of this OOV before overwriting it.
//            current_sent_str[current_sent_str.size()-1] = word;
//            word = Corpus::UNK;
//          }
//        }
//        current_sent.push_back(wordsToInt[word]);
//        current_sent_pos.push_back(posToInt[pos]);
//      }
//    }
//
//    // Add the last sentence.
//    if (current_sent.size() > 0) {
//      current_sent.push_back(wordsToInt[Corpus::ROOT]);
//      current_sent_pos.push_back(posToInt[Corpus::ROOT]);
//      current_sent_str.push_back("");
//      sentencesTest[sentence] = current_sent;
//      sentencesPosTest[sentence] = current_sent_pos;
//      sentencesStrTest[sentence] = current_sent_str;
//      sentencesLemmaTest[sentence] = current_sent_lemma;
//      sentencesXPosTest[sentence] = current_sent_xpos;
//      sentencesFeatTest[sentence] = current_sent_feat;
//      sentencesMultTest[sentence] = current_sent_mult;
//      //sentencesCommTest[sentence] = current_sent_comm;
//      sentence++;
//      nsentencesTest = sentence;
//    }
//
//    actionsFile.close();
//    std::cerr << "test sentence = " << nsentencesTest << std::endl;
//    if (DEBUG){
//      std::cerr<<"done"<<"\n";
//      for (auto a: actions) std::cerr<<a<<"\n";
//    }
//    nactions=actions.size();
//    if (DEBUG){
//      std::cerr<<"nactions:"<<nactions<<"\n";
//      std::cerr<<"nwords:"<<nwords<<"\n";
//      for (unsigned i=0;i<npos;i++) std::cerr<<i<<":"<<intToPos[i]<<"\n";
//    }
//    nactions=actions.size();
//  }

inline void load_correct_actions(std::string file){
	
  std::ifstream actionsFile(file);
  //correct_act_sent=new vector<vector<unsigned>>();
  if (!actionsFile){
    std::cerr << "### File does not exist! ###" << std::endl;
  }
  std::string lineS;
	
  int count=-1;
  int sentence=-1;
  bool initial=false;
  bool first=true;
  wordsToInt[Corpus::BAD0] = 0;
  intToWords[0] = Corpus::BAD0;
  wordsToInt[Corpus::UNK] = 1; // unknown symbol
  intToWords[1] = Corpus::UNK;
  assert(max == 0);
  assert(maxPos == 0);
  max=2;
  maxPos=1;
  
  charsToInt[BAD0]=1;
  intToChars[1]="BAD0";
  maxChars=1;
  
	std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  while (getline(actionsFile, lineS)){
    //istringstream iss(line);
    //string lineS;
 		//iss>>lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
		if (lineS.empty()) {
			count = 0;
			if (!first) {
				sentences[sentence] = current_sent;
				sentencesPos[sentence] = current_sent_pos;
      }
      
			sentence++;
			nsentences = sentence;
      
			initial = true;
                   current_sent.clear();
			current_sent_pos.clear();
		} else if (count == 0) {
			first = false;
			//stack and buffer, for now, leave it like this.
			count = 1;
			if (initial) {
        // the initial line in each sentence may look like:
        // [][][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        if (transition_system == "swap")
          lineS = lineS.substr(3, lineS.size() - 4); // 5, 6 for list-based , 3, 4 for swap
        else
          lineS = lineS.substr(5, lineS.size() - 6);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);
        do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          if (posIndex == std::string::npos) {
            std::cerr << "cant find the dash in '" << word << "'" << std::endl;
          }
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }

          // new word
          if (wordsToInt[word] == 0) {
            wordsToInt[word] = max;
            intToWords[max] = word;
            nwords = max;
            max++;

            unsigned j = 0;
            while(j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              if (charsToInt[wj] == 0) {
                charsToInt[wj] = maxChars;
                intToChars[maxChars] = wj;
                maxChars++;
              }
              j += UTF8Len(word[j]);
            }
          }
        
          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
			}
			initial=false;
		}
		else if (count==1){
			int i=0;
			bool found=false;
			for (auto a: actions) {
				if (a==lineS) {
					std::vector<unsigned> a=correct_act_sent[sentence];
	                                a.push_back(i);
        	                        correct_act_sent[sentence]=a;
					found=true;
				}
				i++;
			}
			if (!found) {
				actions.push_back(lineS);
				std::vector<unsigned> a=correct_act_sent[sentence];
				a.push_back(actions.size()-1);
				correct_act_sent[sentence]=a;
			}
			count=0;
		}
	}

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;
    sentence++;
    nsentences = sentence;
  }
      
  actionsFile.close();
/*	std::string oov="oov";
	posToInt[oov]=maxPos;
        intToPos[maxPos]=oov;
        npos=maxPos;
        maxPos++;
        wordsToInt[oov]=max;
        intToWords[max]=oov;
        nwords=max;
        max++;*/
  if (DEBUG){
	  std::cerr<<"done"<<"\n";
	  for (auto a: actions) {
		  std::cerr<<a<<"\n";
	  }
  }
	nactions=actions.size();
  if (DEBUG){
	std::cerr<<"nactions:"<<nactions<<"\n";
        std::cerr<<"nwords:"<<nwords<<"\n";
	for (unsigned i=0;i<npos;i++){
                std::cerr<<i<<":"<<intToPos[i]<<"\n";
        }
  }
	nactions=actions.size();
	
}

inline unsigned get_or_add_word(const std::string& word) {
  unsigned& id = wordsToInt[word];
  if (id == 0) {
    id = max;
    intToWords[id] = word;
    nwords = max;
    ++max;
  }
  return id;
}

inline void load_correct_actionsDev(std::string file) {
  std::ifstream actionsFile(file);
  if (!actionsFile){
    std::cerr << "### File does not exist! ###" << std::endl;
  }
  std::string lineS;

  assert(maxPos > 1);
  assert(max > 3);
  int count = -1;
  int sentence = -1;
  bool initial = false;
  bool first = true;
  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  std::vector<std::string> current_sent_str;
  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      // an empty line marks the end of a sentence.
      count = 0;
      if (!first) {
        sentencesDev[sentence] = current_sent;
        sentencesPosDev[sentence] = current_sent_pos;
        sentencesStrDev[sentence] = current_sent_str;
      }
      
      sentence++;
      nsentencesDev = sentence;
      
      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
      current_sent_str.clear(); 
    } else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
        // the initial line in each sentence may look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        if (transition_system == "swap")
          lineS = lineS.substr(3, lineS.size() - 4); // 5, 6 for list-based , 3, 4 for swap
        else
          lineS = lineS.substr(5, lineS.size() - 6);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);  
	do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }
          // add an empty string for any token except OOVs (it is easy to 
          // recover the surface form of non-OOV using intToWords(id)).
          current_sent_str.push_back(word);
          // OOV word
          if (wordsToInt[word] == 0) {
//            if (USE_SPELLING) {
//              max = nwords + 1;
//              //std::cerr<< "max:" << max << "\n";
//              wordsToInt[word] = max;
//              intToWords[max] = word;
//              nwords = max;
//            } else {
              // save the surface form of this OOV before overwriting it.
//              current_sent_str[current_sent_str.size()-1] = word;
              word = Corpus::UNK;
//            }
          }
          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
      }
      initial = false;
    } else if (count == 1) {
      auto actionIter = std::find(actions.begin(), actions.end(), lineS);
      if (actionIter != actions.end()) {
        unsigned actionIndex = std::distance(actions.begin(), actionIter);
        correct_act_sentDev[sentence].push_back(actionIndex);
      } else {
        // TODO: right now, new actions which haven't been observed in training
        // are not added to correct_act_sentDev. This may be a problem if the
        // training data is little.
          // new action
          actions.push_back(lineS);
          unsigned actionIndex = actions.size() - 1;
          correct_act_sentDev[sentence].push_back(actionIndex);
      }
      count=0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentencesDev[sentence] = current_sent;
    sentencesPosDev[sentence] = current_sent_pos;
    sentencesStrDev[sentence] = current_sent_str;
    sentence++;
    nsentencesDev = sentence;
  }
  
  actionsFile.close();
}

void ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

};

} // namespace

#endif // __LSTMSDPARSER_CPYPDICT_H__
