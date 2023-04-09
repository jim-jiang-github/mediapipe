#include "google/protobuf/compiler/code_generator.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/printer.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "schema_proto2_to_proto3_util.h"

#include "google/protobuf/compiler/plugin.h"

using google::protobufx::FileDescriptorProto;
using google::protobufx::FileDescriptor;
using google::protobufx::DescriptorPool;
using google::protobufx::io::Printer;
using google::protobufx::util::SchemaGroupStripper;
using google::protobufx::util::EnumScrubber;

namespace google {
namespace protobufx {
namespace compiler {

namespace {

string StripProto(string filename) {
  if (filename.substr(filename.size() - 11) == ".protodevel") {
    // .protodevel
    return filename.substr(0, filename.size() - 11);
  } else {
    // .proto
    return filename.substr(0, filename.size() - 6);
  }
}

DescriptorPool new_pool_;

}  // namespace

class GoGoProtoGenerator : public CodeGenerator {
 public:
  virtual bool GenerateAll(const std::vector<const FileDescriptor*>& files,
                           const string& parameter,
                           GeneratorContext* context,
                           string* error) const {
    for (int i = 0; i < files.size(); i++) {
      for (auto file : files) {
        bool can_generate =
            (new_pool_.FindFileByName(file->name()) == nullptr);
        for (int j = 0; j < file->dependency_count(); j++) {
          can_generate &= (new_pool_.FindFileByName(
              file->dependency(j)->name()) != nullptr);
        }
        for (int j = 0; j < file->public_dependency_count(); j++) {
          can_generate &= (new_pool_.FindFileByName(
              file->public_dependency(j)->name()) != nullptr);
        }
        for (int j = 0; j < file->weak_dependency_count(); j++) {
          can_generate &= (new_pool_.FindFileByName(
              file->weak_dependency(j)->name()) != nullptr);
        }
        if (can_generate) {
          Generate(file, parameter, context, error);
          break;
        }
      }
    }

    return true;
  }

  virtual bool Generate(const FileDescriptor* file,
                        const string& parameter,
                        GeneratorContext* context,
                        string* error) const {
    FileDescriptorProto new_file;
    file->CopyTo(&new_file);
    SchemaGroupStripper::StripFile(file, &new_file);

    EnumScrubber enum_scrubber;
    enum_scrubber.ScrubFile(&new_file);

    string filename = file->name();
    string basename = StripProto(filename);

    std::vector<std::pair<string,string>> option_pairs;
    ParseGeneratorParameter(parameter, &option_pairs);

    std::unique_ptr<google::protobufx::io::ZeroCopyOutputStream> output(
        context->Open(basename + ".proto"));
    string content = new_pool_.BuildFile(new_file)->DebugString();
    Printer printer(output.get(), '$');
    printer.WriteRaw(content.c_str(), content.size());

    return true;
  }
};

}  // namespace compiler
}  // namespace protobufx
}  // namespace google

int main(int argc, char* argv[]) {
  google::protobufx::compiler::GoGoProtoGenerator generator;
  return google::protobufx::compiler::PluginMain(argc, argv, &generator);
}
