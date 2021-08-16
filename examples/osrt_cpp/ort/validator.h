#ifndef _VALIDATOR_H
#define _VALIDATOR_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class Validator {
private:
    // ORT Session
    Ort::Session _session;

    // Input information
    size_t _num_input_nodes;
    std::vector<const char*> _input_node_names;
    std::vector<int64_t> _input_node_dims;

    int _image_size;
    std::string _image_path;
    std::string _labels_path;

    void PrepareInputs();
    void ScoreModel();
    void Validate();
    
    std::vector<std::string> ReadFileToVec(std::string fname);

public:
    int GetImageSize() const;
    Validator(Ort::Env& env, std::string model_path, std::string image_path, std::string labels_path,
              Ort::SessionOptions& session_options);
};

#endif // _VALIDATOR_H
