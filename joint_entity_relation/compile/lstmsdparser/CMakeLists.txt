include_directories (${PROJECT_SOURCE_DIR}/)

set (lstm_sdparser_VERSION "0.0.1")
set (lstm_sdparser_SRC
	lstm_sdparser.cpp
    layers.cc
    theirtreelstm.cc
    swapbased.cc
    listbased.cc
    )

add_executable(lstmparse main.cpp ${lstm_sdparser_SRC})
target_link_libraries(lstmparse dynet ${Boost_LIBRARIES})