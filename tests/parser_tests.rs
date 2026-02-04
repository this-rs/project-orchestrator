//! Parser unit tests
//!
//! These tests don't require external services.
//! Run with: cargo test --test parser_tests

use project_orchestrator::parser::CodeParser;
use std::path::Path;

#[test]
fn test_parser_creation() {
    let parser = CodeParser::new();
    assert!(parser.is_ok(), "Parser should initialize");
}

#[test]
fn test_parse_rust_function() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
/// A simple greeting function
pub fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code: {:?}", result.err());

    let parsed = result.unwrap();
    assert_eq!(parsed.language, "rust");
    assert!(!parsed.functions.is_empty(), "Should find functions");

    let func = &parsed.functions[0];
    assert_eq!(func.name, "hello");
    assert!(!func.params.is_empty(), "Should have parameters");
    assert!(func.return_type.is_some(), "Should have return type");
}

#[test]
fn test_parse_rust_struct() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
/// A person struct
pub struct Person {
    pub name: String,
    age: u32,
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.structs.is_empty(), "Should find structs");

    let s = &parsed.structs[0];
    assert_eq!(s.name, "Person");
}

#[test]
fn test_parse_rust_enum() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub enum Status {
    Pending,
    InProgress,
    Completed,
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.enums.is_empty(), "Should find enums");

    let e = &parsed.enums[0];
    assert_eq!(e.name, "Status");
    assert_eq!(e.variants.len(), 3);
}

#[test]
fn test_parse_rust_imports() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

fn main() {}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.imports.is_empty(), "Should find imports");
}

#[test]
fn test_parse_rust_impl_methods() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
struct Calculator;

impl Calculator {
    pub fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }

    fn subtract(&self, a: i32, b: i32) -> i32 {
        a - b
    }
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    // Should find struct and methods
    assert!(!parsed.structs.is_empty(), "Should find struct");
    assert!(parsed.functions.len() >= 2, "Should find impl methods");
}

#[test]
fn test_parse_rust_async_function() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub async fn fetch_data(url: &str) -> Result<String, Error> {
    todo!()
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.functions.is_empty(), "Should find function");

    let func = &parsed.functions[0];
    assert!(func.is_async, "Function should be marked as async");
}

#[test]
fn test_parse_typescript() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}

class UserService {
    async getUser(id: string): Promise<User> {
        return { name: "Test", age: 25 };
    }
}
"#;

    let path = Path::new("test.ts");
    let result = parser.parse_file(path, code);

    assert!(
        result.is_ok(),
        "Should parse TypeScript code: {:?}",
        result.err()
    );

    let parsed = result.unwrap();
    assert_eq!(parsed.language, "typescript");
    assert!(!parsed.functions.is_empty(), "Should find functions");
    assert!(!parsed.structs.is_empty(), "Should find classes/interfaces");
}

#[test]
fn test_parse_python() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
import os
from typing import List

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

def main():
    calc = Calculator()
    print(calc.add(1, 2))

async def fetch_data(url: str) -> str:
    pass
"#;

    let path = Path::new("test.py");
    let result = parser.parse_file(path, code);

    assert!(
        result.is_ok(),
        "Should parse Python code: {:?}",
        result.err()
    );

    let parsed = result.unwrap();
    assert_eq!(parsed.language, "python");
    assert!(!parsed.functions.is_empty(), "Should find functions");
    assert!(!parsed.structs.is_empty(), "Should find classes");
    assert!(!parsed.imports.is_empty(), "Should find imports");
}

#[test]
fn test_parse_go() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func (p *Person) Greet() string {
    return fmt.Sprintf("Hello, %s!", p.Name)
}

func main() {
    p := Person{Name: "Alice", Age: 30}
    fmt.Println(p.Greet())
}
"#;

    let path = Path::new("test.go");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Go code: {:?}", result.err());

    let parsed = result.unwrap();
    assert_eq!(parsed.language, "go");
    assert!(!parsed.functions.is_empty(), "Should find functions");
    assert!(!parsed.structs.is_empty(), "Should find structs");
}

#[test]
fn test_unsupported_extension() {
    let mut parser = CodeParser::new().unwrap();

    let code = "some content";
    let path = Path::new("test.xyz");
    let result = parser.parse_file(path, code);

    assert!(result.is_err(), "Should fail for unsupported extension");
}

#[test]
fn test_complexity_calculation() {
    let mut parser = CodeParser::new().unwrap();

    // Simple function - complexity 1
    let simple = r#"
fn simple() -> i32 {
    42
}
"#;

    // Complex function - higher complexity
    let complex = r#"
fn complex(x: i32) -> i32 {
    if x > 0 {
        if x > 10 {
            100
        } else {
            50
        }
    } else {
        for i in 0..10 {
            if i == x {
                return i;
            }
        }
        0
    }
}
"#;

    let path = Path::new("test.rs");

    let simple_parsed = parser.parse_file(path, simple).unwrap();
    let complex_parsed = parser.parse_file(path, complex).unwrap();

    let simple_complexity = simple_parsed.functions[0].complexity;
    let complex_complexity = complex_parsed.functions[0].complexity;

    assert!(
        complex_complexity > simple_complexity,
        "Complex function should have higher complexity"
    );
}

#[test]
fn test_hash_generation() {
    let mut parser = CodeParser::new().unwrap();

    let code1 = "fn test1() {}";
    let code2 = "fn test2() {}";

    let path = Path::new("test.rs");

    let parsed1 = parser.parse_file(path, code1).unwrap();
    let parsed2 = parser.parse_file(path, code2).unwrap();

    assert_ne!(
        parsed1.hash, parsed2.hash,
        "Different code should have different hashes"
    );

    // Same code should have same hash
    let parsed1_again = parser.parse_file(path, code1).unwrap();
    assert_eq!(
        parsed1.hash, parsed1_again.hash,
        "Same code should have same hash"
    );
}

#[test]
fn test_parse_rust_trait() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
/// A trait for displayable items
pub trait Display {
    fn display(&self) -> String;
    fn format(&self, prefix: &str) -> String;
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.traits.is_empty(), "Should find traits");

    let t = &parsed.traits[0];
    assert_eq!(t.name, "Display");
}

#[test]
fn test_parse_rust_impl_block() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
struct MyStruct;

trait MyTrait {
    fn do_something(&self);
}

impl MyStruct {
    fn new() -> Self {
        MyStruct
    }
}

impl MyTrait for MyStruct {
    fn do_something(&self) {
        println!("Done!");
    }
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();

    // Should find struct, trait, and impl blocks
    assert!(!parsed.structs.is_empty(), "Should find struct");
    assert!(!parsed.traits.is_empty(), "Should find trait");
    assert!(!parsed.impl_blocks.is_empty(), "Should find impl blocks");

    // Should have 2 impl blocks
    assert_eq!(parsed.impl_blocks.len(), 2, "Should find 2 impl blocks");

    // One impl should be for a trait
    let trait_impl = parsed.impl_blocks.iter().find(|i| i.trait_name.is_some());
    assert!(trait_impl.is_some(), "Should have a trait impl");
    assert_eq!(trait_impl.unwrap().trait_name, Some("MyTrait".to_string()));
}

#[test]
fn test_parse_function_calls() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
fn helper() -> i32 {
    42
}

fn another_helper(x: i32) -> i32 {
    x * 2
}

fn main() {
    let a = helper();
    let b = another_helper(a);
    println!("{}", b);
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();

    // Should find function calls
    assert!(
        !parsed.function_calls.is_empty(),
        "Should find function calls"
    );

    // The main function should call helper and another_helper
    let helper_calls: Vec<_> = parsed
        .function_calls
        .iter()
        .filter(|c| c.callee_name == "helper" || c.callee_name == "another_helper")
        .collect();

    assert!(
        helper_calls.len() >= 2,
        "Should find calls to helper and another_helper"
    );
}

#[test]
fn test_parse_method_calls() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
struct Counter {
    value: i32,
}

impl Counter {
    fn increment(&mut self) {
        self.value += 1;
    }

    fn get(&self) -> i32 {
        self.value
    }
}

fn main() {
    let mut counter = Counter { value: 0 };
    counter.increment();
    let val = counter.get();
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();

    // Should find method calls
    let method_calls: Vec<_> = parsed
        .function_calls
        .iter()
        .filter(|c| c.callee_name == "increment" || c.callee_name == "get")
        .collect();

    assert!(
        method_calls.len() >= 2,
        "Should find method calls to increment and get"
    );
}

#[test]
fn test_parse_rust_generic_struct() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub struct Container<T, U> {
    first: T,
    second: U,
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.structs.is_empty(), "Should find struct");

    let s = &parsed.structs[0];
    assert_eq!(s.name, "Container");
    assert_eq!(
        s.generics.len(),
        2,
        "Should have 2 type parameters, got: {:?}",
        s.generics
    );
    assert!(s.generics.contains(&"T".to_string()), "Should contain T");
    assert!(s.generics.contains(&"U".to_string()), "Should contain U");
}

#[test]
fn test_parse_rust_generic_function() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub fn transform<T, U>(input: T) -> U {
    todo!()
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.functions.is_empty(), "Should find function");

    let func = &parsed.functions[0];
    assert_eq!(func.name, "transform");
    assert_eq!(func.generics.len(), 2, "Should have 2 type parameters");
    assert!(func.generics.contains(&"T".to_string()), "Should contain T");
    assert!(func.generics.contains(&"U".to_string()), "Should contain U");
}

#[test]
fn test_parse_rust_generic_trait() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub trait Converter<From, To> {
    fn convert(&self, value: From) -> To;
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.traits.is_empty(), "Should find trait");

    let t = &parsed.traits[0];
    assert_eq!(t.name, "Converter");
    assert_eq!(t.generics.len(), 2, "Should have 2 type parameters");
    assert!(
        t.generics.contains(&"From".to_string()),
        "Should contain From"
    );
    assert!(t.generics.contains(&"To".to_string()), "Should contain To");
}

#[test]
fn test_parse_rust_generic_impl() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn new(value: T) -> Self {
        Wrapper { value }
    }
}

impl<T: Clone> Clone for Wrapper<T> {
    fn clone(&self) -> Self {
        Wrapper { value: self.value.clone() }
    }
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert_eq!(parsed.impl_blocks.len(), 2, "Should find 2 impl blocks");

    // Find the inherent impl (no trait)
    let inherent_impl = parsed
        .impl_blocks
        .iter()
        .find(|i| i.trait_name.is_none())
        .expect("Should have an inherent impl");
    assert_eq!(inherent_impl.for_type, "Wrapper");
    assert!(
        !inherent_impl.generics.is_empty(),
        "Inherent impl should have generics"
    );
    assert!(
        inherent_impl.generics.iter().any(|g| g.contains('T')),
        "Should have T parameter"
    );

    // Find the trait impl
    let trait_impl = parsed
        .impl_blocks
        .iter()
        .find(|i| i.trait_name.is_some())
        .expect("Should have a trait impl");
    assert_eq!(trait_impl.for_type, "Wrapper");
    assert_eq!(trait_impl.trait_name, Some("Clone".to_string()));
    assert!(
        !trait_impl.generics.is_empty(),
        "Trait impl should have generics"
    );
    // The constrained type parameter should include the bound
    assert!(
        trait_impl.generics.iter().any(|g| g.contains("Clone")),
        "Should have Clone-bounded parameter: {:?}",
        trait_impl.generics
    );
}

#[test]
fn test_parse_rust_lifetime_parameter() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub struct Ref<'a, T> {
    value: &'a T,
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.structs.is_empty(), "Should find struct");

    let s = &parsed.structs[0];
    assert_eq!(s.name, "Ref");
    assert!(
        s.generics.len() >= 2,
        "Should have lifetime and type parameter"
    );
    assert!(
        s.generics.iter().any(|g| g.contains("'a")),
        "Should contain 'a lifetime"
    );
    assert!(s.generics.contains(&"T".to_string()), "Should contain T");
}

#[test]
fn test_parse_rust_constrained_generics() {
    let mut parser = CodeParser::new().unwrap();

    let code = r#"
pub fn process<T: Clone + Send, U: Default>(input: T) -> U {
    todo!()
}
"#;

    let path = Path::new("test.rs");
    let result = parser.parse_file(path, code);

    assert!(result.is_ok(), "Should parse Rust code");

    let parsed = result.unwrap();
    assert!(!parsed.functions.is_empty(), "Should find function");

    let func = &parsed.functions[0];
    assert_eq!(func.name, "process");
    assert_eq!(func.generics.len(), 2, "Should have 2 type parameters");
    // Constrained generics should include the bounds
    assert!(
        func.generics.iter().any(|g| g.contains("Clone")),
        "Should contain Clone bound"
    );
    assert!(
        func.generics.iter().any(|g| g.contains("Default")),
        "Should contain Default bound"
    );
}
