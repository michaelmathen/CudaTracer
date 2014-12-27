
#include <iostream>
#include "ParsingException.hpp"

namespace mm_ray {

  
  void ParsingException::clear(){
    _error_msgs.clear();
  }
  static const char* kTypeNames[] = 
    { "Null", "False", "True", "Object", "Array", "String", "Number" };
  
  const char* ParsingException::what() {
    return _error_msgs.str().c_str();
  }

  
  void ParsingException::checkType(rapidjson::Value& val, ValueType type) {
    switch(type){
    case ValueType::VAL_DOUBLE:
      if (!val.IsDouble()) {
	_error_msgs << "Value is not double\n";
	WriteContext(val);
	throw this;
      }
      break;
    case ValueType::VAL_INT:
      if (!val.IsInt()) {
      	_error_msgs << "Value is not a int\n";
	WriteContext(val);
	throw this;
      }
      break;
    case ValueType::VAL_UINT:
      if (!val.IsUint()) {
	_error_msgs << "Value is not an unsigned integer\n";
	WriteContext(val);
	_error_msgs << " to uint failed\n";
	throw this;
      }
      break;
    case ValueType::VAL_ARRAY:
      if (!val.IsArray()) {
	_error_msgs << "Value is not a array\n";
	WriteContext(val);
	throw this;
      }
      break;
    case ValueType::VAL_BOOLEAN:
      if (!val.IsBool()) {
	_error_msgs << "Value is not a bool\n";
	WriteContext(val);
	throw this;
      }
      break;
    case ValueType::VAL_NULL:
      if (!val.IsNull()) {
	_error_msgs << "Value is not null\n" ;
	WriteContext(val);
	throw this;
      }      
      break;
    case ValueType::VAL_STRING:
      if (!val.IsString()) {
	_error_msgs << "Value is not string\n";
	WriteContext(val);
	throw this;
      }

    }
  }
  void ParsingException::WriteContext(rapidjson::Value& val){
    /*
      Print the tree of the json file that we messed up on.
     */
    if (val.IsObject()){
      _error_msgs << 	"{\n";
      for (auto it = val.MemberBegin(); it != val.MemberEnd(); ++it){

	_error_msgs << it->name.GetString()
		    << ":";
	WriteContext(it->value);
	_error_msgs << " ";
      }
      _error_msgs << 	"},\n";
    } else if (val.IsArray()){
      _error_msgs << "[\n";

      for (auto it = val.Begin(); it != val.End(); ++it){
	WriteContext(*it);
	_error_msgs << " ";
      }
      _error_msgs << "\n],\n";
    } else if (val.IsInt()) {
      _error_msgs << val.GetInt() << " : int";
    } else if (val.IsUint()) {
      _error_msgs << val.GetUint() << " : uint";
    } else if (val.IsBool()) {
      _error_msgs << val.GetBool() << " : bool";
    } else if (val.IsDouble()) {
      _error_msgs << val.GetDouble() << " : double";
    } else if (val.IsInt64()) {
      _error_msgs << val.GetInt64() << " : int64";
    } else if (val.IsNull()) {
      _error_msgs << "NULL" << " : null";
    } else if (val.IsString()) {
      _error_msgs << val.GetString() << " : string";
    } else if (val.IsUint64()) {
      _error_msgs << val.GetUint64() << " : uint64";
    } else {
      _error_msgs << "Unknown value type\n";
    }
    
    
  }
  
  rapidjson::Value& ParsingException::AssertGetMember(rapidjson::Value& val,  string const& member) {
    if (val.HasMember(member.c_str()) == 0){
      _error_msgs << "No member " + member + " around:\n";
      WriteContext(val);
      throw this;
    }

    return val[member.c_str()];
  }


  rapidjson::Value& ParsingException::AssertGetMember(rapidjson::Value& val,  int i) {
    if (i >= val.Size()){
      _error_msgs << "Out of array index " << i << " in json: "<<  val.GetString();
      throw this;
    }

    return val[i];
  }

  template<>
  int ParsingException::AssertIsType<int>(rapidjson::Value& val) {
    checkType(val, ValueType::VAL_INT);
    return val.GetInt();
  }

  template<>
  double ParsingException::AssertIsType<double>(rapidjson::Value& val){
    checkType(val, ValueType::VAL_DOUBLE);
    return val.GetDouble();

  }

  template<>
  float ParsingException::AssertIsType<float>(rapidjson::Value& val){
    checkType(val, ValueType::VAL_DOUBLE);
    return val.GetDouble();
  }

  
  template<>
  string ParsingException::AssertIsType<string>(rapidjson::Value& val){
    checkType(val, ValueType::VAL_STRING);
    return val.GetString();
  }

  template<>
  unsigned int ParsingException::AssertIsType<unsigned int>(rapidjson::Value& val){
    parse_err.checkType(val, ValueType::VAL_UINT);
    return val.GetUint();
  }

}


