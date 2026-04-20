#include "trtf/config/schema_registry.h"

#include <algorithm>
#include <stdexcept>

namespace trtf::config {

SchemaRegistry& SchemaRegistry::instance()
{
    static SchemaRegistry registry;
    return registry;
}

void SchemaRegistry::register_schema(Schema schema)
{
    if (schema.namespace_name.empty())
        throw std::invalid_argument("Cannot register schema with empty namespace");
    if (schema.fields.empty())
    {
        throw std::invalid_argument(
            "Cannot register schema with no fields for namespace: " + schema.namespace_name);
    }
    for (const auto& field : schema.fields)
    {
        if (field.name.empty())
        {
            throw std::invalid_argument(
                "Field with empty name in namespace: " + schema.namespace_name);
        }
        if (field.allowed_layers.count(Layer::SchemaDefault) != 0)
        {
            // SchemaDefault is the fallback, not a contribution. Rejecting at
            // registration surfaces schema authoring mistakes at static-init.
            throw std::invalid_argument(
                "Field " + schema.namespace_name + "." + field.name +
                " declares SchemaDefault in allowed_layers; that layer is reserved for "
                "the baked-in default and cannot be contributed.");
        }
        if (field.allowed_layers.empty())
        {
            throw std::invalid_argument(
                "Field " + schema.namespace_name + "." + field.name +
                " has empty allowed_layers; at least one layer must be permitted "
                "(otherwise the field is unreachable).");
        }
    }
    if (schemas_.count(schema.namespace_name) != 0)
    {
        // Duplicate namespace = two agents claimed the same name. Crash at
        // static-init so the collision is visible immediately.
        throw std::invalid_argument(
            "Duplicate config schema for namespace: " + schema.namespace_name);
    }
    schemas_[schema.namespace_name] = std::move(schema);
}

const Schema* SchemaRegistry::lookup(const std::string& namespace_name) const
{
    auto it = schemas_.find(namespace_name);
    return (it != schemas_.end()) ? &it->second : nullptr;
}

std::vector<std::string> SchemaRegistry::registered_namespaces() const
{
    std::vector<std::string> names;
    names.reserve(schemas_.size());
    for (const auto& kv : schemas_)
        names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

void SchemaRegistry::clear_for_testing()
{
    schemas_.clear();
}

} // namespace trtf::config
