#include "../include/supportLib.hpp"

using namespace std;

unsigned char *DoubleArrayToByteArray(vector<double> *data){
	unsigned char *out;
	size_t i;

	out = new unsigned char[data->size()];

	for(i = 0; i < data->size(); i++){
		out[i] = data->at(i);
	}

	return out;
}

void WriteToFile(vector<double> *data, string filename){
	unsigned char *bytes;

	bytes = DoubleArrayToByteArray(data);

	ofstream file(filename.c_str(), ios::binary);
	file.write(reinterpret_cast<char *>(bytes), data->size());
	file.close();

	delete bytes;
}

vector<double> *ByteArrayToDoubleArray(vector<unsigned char> *data){
	vector<double> *out;
	size_t i;

	out = new vector<double>(data->size());

	for(i = 0; i < data->size(); i++){
		out->at(i) = data->at(i);
	}

	return out;
}