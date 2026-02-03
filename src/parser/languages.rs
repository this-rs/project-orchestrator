//! Language-specific utilities and queries

/// Tree-sitter query patterns for different languages
pub mod queries {
    /// Rust query patterns
    pub mod rust {
        pub const FUNCTIONS: &str = r#"
            (function_item
                name: (identifier) @name
                parameters: (parameters) @params
                return_type: (_)? @return_type
                body: (block) @body
            ) @function
        "#;

        pub const STRUCTS: &str = r#"
            (struct_item
                name: (type_identifier) @name
                body: (field_declaration_list)? @fields
            ) @struct
        "#;

        pub const TRAITS: &str = r#"
            (trait_item
                name: (type_identifier) @name
                body: (declaration_list) @body
            ) @trait
        "#;

        pub const IMPORTS: &str = r#"
            (use_declaration
                argument: (_) @path
            ) @import
        "#;

        pub const CALLS: &str = r#"
            (call_expression
                function: [
                    (identifier) @callee
                    (field_expression
                        field: (field_identifier) @callee
                    )
                    (scoped_identifier
                        name: (identifier) @callee
                    )
                ]
            ) @call
        "#;
    }

    /// TypeScript query patterns
    pub mod typescript {
        pub const FUNCTIONS: &str = r#"
            [
                (function_declaration
                    name: (identifier) @name
                    parameters: (formal_parameters) @params
                    body: (statement_block) @body
                )
                (method_definition
                    name: (property_identifier) @name
                    parameters: (formal_parameters) @params
                    body: (statement_block) @body
                )
                (arrow_function
                    parameters: (formal_parameters) @params
                    body: (_) @body
                )
            ] @function
        "#;

        pub const CLASSES: &str = r#"
            (class_declaration
                name: (type_identifier) @name
                body: (class_body) @body
            ) @class
        "#;

        pub const INTERFACES: &str = r#"
            (interface_declaration
                name: (type_identifier) @name
                body: (object_type) @body
            ) @interface
        "#;

        pub const IMPORTS: &str = r#"
            (import_statement
                source: (string) @source
            ) @import
        "#;
    }

    /// Python query patterns
    pub mod python {
        pub const FUNCTIONS: &str = r#"
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                body: (block) @body
            ) @function
        "#;

        pub const CLASSES: &str = r#"
            (class_definition
                name: (identifier) @name
                body: (block) @body
            ) @class
        "#;

        pub const IMPORTS: &str = r#"
            [
                (import_statement
                    name: (dotted_name) @name
                )
                (import_from_statement
                    module_name: (dotted_name) @module
                )
            ] @import
        "#;
    }

    /// Go query patterns
    pub mod go {
        pub const FUNCTIONS: &str = r#"
            [
                (function_declaration
                    name: (identifier) @name
                    parameters: (parameter_list) @params
                    body: (block) @body
                )
                (method_declaration
                    name: (field_identifier) @name
                    parameters: (parameter_list) @params
                    body: (block) @body
                )
            ] @function
        "#;

        pub const STRUCTS: &str = r#"
            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                    type: (struct_type) @body
                )
            ) @struct
        "#;

        pub const INTERFACES: &str = r#"
            (type_declaration
                (type_spec
                    name: (type_identifier) @name
                    type: (interface_type) @body
                )
            ) @interface
        "#;

        pub const IMPORTS: &str = r#"
            (import_declaration
                (import_spec
                    path: (interpreted_string_literal) @path
                )
            ) @import
        "#;
    }
}
