#pragma once

#include <QString>
#include <vector>
#include <functional>

namespace ColourScheme
{
    enum class Type
    {
        NONE = 0,
        COMPLEMENTARY = 1,
        TRIADIC = 2,
        COMPOUND = 3,
        MAX = 4
    };

    static const std::vector<QString> Type_STR = {
        "None",
        "Complementary",
        "Triadic",
        "Compound"
    };

    //Converts type string to type enum
    Type strToEnum(const QString& t_type);

    //Alias for function wrapper
    using FunctionType = std::function<void()>;

    //Returns function wrapper from enum
    FunctionType getFunction(const Type& t_type);
};

