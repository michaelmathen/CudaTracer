
#include <sstream>
#include <exception>
#include "rapidjson/document.h"

/*
  This is for handling error that occur while parsing the json file. 
  The general idea is that we try to give enough information so that 
  we know roughly where the issue is by providing a get method that 
  handles retrieving elements from the file.
 */


namespace mm_ray {
  using namespace std;

  enum class ValueType {
    VAL_DOUBLE,
      VAL_INT,
      VAL_UINT,
      VAL_ARRAY,
      VAL_BOOLEAN,
      VAL_NULL,
      VAL_STRING
      };

  class ParsingException : public exception {

    stringstream _error_msgs;


		   
  public:

    template<typename T>
    ParsingException& operator<<(T const& error_msg){
      _error_msgs << error_msg;
      return *this;
    }
    void checkType(rapidjson::Value& val, ValueType type_name);
    
    template<typename T>
    T AssertIsType(rapidjson::Value& val);
    
    rapidjson::Value& AssertGetMember(rapidjson::Value& val,  string const& member);

    rapidjson::Value& AssertGetMember(rapidjson::Value& val,  int i);

    void WriteContext(rapidjson::Value& val);
    
    virtual void clear();

    virtual const char* what();

    template<typename T>
    T get(rapidjson::Value& val, const char* member){
      if (!val.IsObject()){
	_error_msgs << "Using query: " << member << " on non object:\n";
	WriteContext(val);
	throw this;
      }
      auto& tmp = AssertGetMember(val, member);
      return AssertIsType<T>(tmp);
    }
    
    template<typename T>
    T get(rapidjson::Value& val, int i){

      if (!val.IsArray()){
	_error_msgs << "Using querry: " << i << " on non array\n";
	WriteContext(val);
	throw this;
      }
      auto& tmp = AssertGetMember(val, i);
      return AssertIsType<T>(tmp);
    }
    
    rapidjson::Value& get(rapidjson::Value& val, const char* member){
      if (!val.IsObject()){
	_error_msgs << "Using query: " << member << " on non object:\n";
	WriteContext(val);
	throw this;
      }

      return AssertGetMember(val, member);
    }
    
    rapidjson::Value& get(rapidjson::Value& val, int i){
      if (!val.IsArray()){
	_error_msgs << "Using querry: " << i << " on non array\n";
	WriteContext(val);
	throw this;
      }

      return AssertGetMember(val, i);
    }

  };
  
  static ParsingException parse_err;
}
