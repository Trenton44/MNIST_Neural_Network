#include <vector>
#include <fstream>
#include <iostream>

typedef std::vector<double> Sample;
std::vector<double> parseCSVLine(std::string &str, double normalizer){
    std::vector<double> line;
    std::string tempstr;
    for(unsigned cursor = 0; cursor <str.length(); cursor++){
        if(str[cursor] == ','){
            line.push_back(std::stod(tempstr) / normalizer);
            tempstr.clear();
        }
        else
            tempstr.push_back(str[cursor]);
    }
    return line;
}

void readCSV(std::vector<Sample> &data, std::string filename, double normalizer){
    std::cout << "opening " << filename << std::endl;
    std::fstream file;
    std::string str;
    file.open(filename, std::fstream::in);
    std::cout << "file successfully opened" << std::endl;
    std::cout << "reading dataset into memory." << std::endl;
    unsigned counter = 0;
    while(std::getline(file, str)){
        data.push_back(parseCSVLine(str, normalizer));
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