//
// Created by junfeng on 3/12/16.
//
#include <string>
#include <iostream>
#include <locale>
#include <codecvt>
#include <sstream>

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>

struct MyClass {
    std::u32string u32;

    // This method lets cereal know which data members to serialize
    template<class Archive>
    void serialize(Archive &archive) {
        archive(u32); // serialize things by passing them to the archive
    }
};


int main() {
    std::stringstream ss; // any stream can be used

    {
        cereal::BinaryOutputArchive oarchive(ss); // Create an output archive

        MyClass m1;
        m1.u32 = U"你好 cereal";

        oarchive(m1); // Write the data to the archive
    }

    {
        cereal::BinaryInputArchive iarchive(ss); // Create an input archive

        MyClass m2;
        iarchive(m2); // Read the data from the archive
        using namespace std;
        wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
        std::cout << conv.to_bytes(m2.u32) << endl;
    }
}