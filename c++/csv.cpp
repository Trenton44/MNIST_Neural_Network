#include <vector>
#include <fstream>
#include <iostream>

double MAX_PIXEL_VALUE = 255.0;
typedef std::vector<double> Sample;
std::vector<double> parseCSVLine(std::string &str){
    std::vector<double> line;
    std::string tempstr;
    for(unsigned cursor = 0; cursor <str.length();  cursor++){
        if(str[cursor] == ','){
            line.push_back((double)atoi(tempstr.c_str()) / MAX_PIXEL_VALUE);
            tempstr.clear();
        }
        else
            tempstr.push_back(str[cursor]);
    }
    return line;
}

void readCSV(std::vector<Sample> &data, std::string filename){
    std::cout << "opening " << filename << std::endl;
    std::fstream file;
    std::string str;
    file.open(filename, std::fstream::in);
    std::cout << "file successfully opened" << std::endl;
    std::cout << "reading dataset into memory." << std::endl;
    unsigned counter = 0;
    while(std::getline(file, str)){
        data.push_back(parseCSVLine(str));
        str.clear();
        counter += 1;
    }
    std::cout << filename << " successfully loaded " << counter << " samples." << std::endl;
    file.close();
    std::cout << "Closed " << filename << std::endl;
}

void parseDataset(std::vector<Sample> &data, std::vector<double> &target_values){
    std::cout<< "parsing csv dataset." << std::endl;
    for(unsigned i = 0; i < data.size(); i++){
        target_values.push_back(data[i].front());
        data[i].erase(data[i].begin());
    }
    std::cout << "successfully parsed dataset. " << std::endl;
}