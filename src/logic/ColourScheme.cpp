#include "ColourScheme.h"
#include <stdexcept>

//Converts type string to type enum
ColourScheme::Type ColourScheme::strToEnum(const QString& t_type)
{
    for (size_t i = 0; i < static_cast<size_t>(Type::MAX); ++i)
    {
        if (Type_STR.at(i).compare(t_type) == 0)
            return static_cast<Type>(i);
    }

    return Type::MAX;
}

//Returns function wrapper from enum
ColourScheme::FunctionType ColourScheme::getFunction(const Type& t_type)
{
    switch (t_type)
    {
    case Type::NONE: return []() { return; };
    default: throw std::invalid_argument(Q_FUNC_INFO " No function for given type");
    }
}
