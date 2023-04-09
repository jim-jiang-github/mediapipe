// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: bind_front.h
// -----------------------------------------------------------------------------
//
// `abslx::bind_front()` returns a functor by binding a number of arguments to
// the front of a provided (usually more generic) functor. Unlike `std::bind`,
// it does not require the use of argument placeholders. The simpler syntax of
// `abslx::bind_front()` allows you to avoid known misuses with `std::bind()`.
//
// `abslx::bind_front()` is meant as a drop-in replacement for C++20's upcoming
// `std::bind_front()`, which similarly resolves these issues with
// `std::bind()`. Both `bind_front()` alternatives, unlike `std::bind()`, allow
// partial function application. (See
// https://en.wikipedia.org/wiki/Partial_application).

#ifndef ABSL_FUNCTIONAL_BIND_FRONT_H_
#define ABSL_FUNCTIONAL_BIND_FRONT_H_

#include "absl/functional/internal/front_binder.h"
#include "absl/utility/utility.h"

namespace abslx {
ABSL_NAMESPACE_BEGIN

// bind_front()
//
// Binds the first N arguments of an invocable object and stores them by value.
//
// Like `std::bind()`, `abslx::bind_front()` is implicitly convertible to
// `std::function`.  In particular, it may be used as a simpler replacement for
// `std::bind()` in most cases, as it does not require  placeholders to be
// specified. More importantly, it provides more reliable correctness guarantees
// than `std::bind()`; while `std::bind()` will silently ignore passing more
// parameters than expected, for example, `abslx::bind_front()` will report such
// mis-uses as errors.
//
// abslx::bind_front(a...) can be seen as storing the results of
// std::make_tuple(a...).
//
// Example: Binding a free function.
//
//   int Minus(int a, int b) { return a - b; }
//
//   assert(abslx::bind_front(Minus)(3, 2) == 3 - 2);
//   assert(abslx::bind_front(Minus, 3)(2) == 3 - 2);
//   assert(abslx::bind_front(Minus, 3, 2)() == 3 - 2);
//
// Example: Binding a member function.
//
//   struct Math {
//     int Double(int a) const { return 2 * a; }
//   };
//
//   Math math;
//
//   assert(abslx::bind_front(&Math::Double)(&math, 3) == 2 * 3);
//   // Stores a pointer to math inside the functor.
//   assert(abslx::bind_front(&Math::Double, &math)(3) == 2 * 3);
//   // Stores a copy of math inside the functor.
//   assert(abslx::bind_front(&Math::Double, math)(3) == 2 * 3);
//   // Stores std::unique_ptr<Math> inside the functor.
//   assert(abslx::bind_front(&Math::Double,
//                           std::unique_ptr<Math>(new Math))(3) == 2 * 3);
//
// Example: Using `abslx::bind_front()`, instead of `std::bind()`, with
//          `std::function`.
//
//   class FileReader {
//    public:
//     void ReadFileAsync(const std::string& filename, std::string* content,
//                        const std::function<void()>& done) {
//       // Calls Executor::Schedule(std::function<void()>).
//       Executor::DefaultExecutor()->Schedule(
//           abslx::bind_front(&FileReader::BlockingRead, this,
//                            filename, content, done));
//     }
//
//    private:
//     void BlockingRead(const std::string& filename, std::string* content,
//                       const std::function<void()>& done) {
//       CHECK_OK(file::GetContents(filename, content, {}));
//       done();
//     }
//   };
//
// `abslx::bind_front()` stores bound arguments explicitly using the type passed
// rather than implicitly based on the type accepted by its functor.
//
// Example: Binding arguments explicitly.
//
//   void LogStringView(abslx::string_view sv) {
//     LOG(INFO) << sv;
//   }
//
//   Executor* e = Executor::DefaultExecutor();
//   std::string s = "hello";
//   abslx::string_view sv = s;
//
//   // abslx::bind_front(LogStringView, arg) makes a copy of arg and stores it.
//   e->Schedule(abslx::bind_front(LogStringView, sv)); // ERROR: dangling
//                                                     // string_view.
//
//   e->Schedule(abslx::bind_front(LogStringView, s));  // OK: stores a copy of
//                                                     // s.
//
// To store some of the arguments passed to `abslx::bind_front()` by reference,
//  use std::ref()` and `std::cref()`.
//
// Example: Storing some of the bound arguments by reference.
//
//   class Service {
//    public:
//     void Serve(const Request& req, std::function<void()>* done) {
//       // The request protocol buffer won't be deleted until done is called.
//       // It's safe to store a reference to it inside the functor.
//       Executor::DefaultExecutor()->Schedule(
//           abslx::bind_front(&Service::BlockingServe, this, std::cref(req),
//           done));
//     }
//
//    private:
//     void BlockingServe(const Request& req, std::function<void()>* done);
//   };
//
// Example: Storing bound arguments by reference.
//
//   void Print(const std::string& a, const std::string& b) {
//     std::cerr << a << b;
//   }
//
//   std::string hi = "Hello, ";
//   std::vector<std::string> names = {"Chuk", "Gek"};
//   // Doesn't copy hi.
//   for_each(names.begin(), names.end(),
//            abslx::bind_front(Print, std::ref(hi)));
//
//   // DO NOT DO THIS: the functor may outlive "hi", resulting in
//   // dangling references.
//   foo->DoInFuture(abslx::bind_front(Print, std::ref(hi), "Guest"));  // BAD!
//   auto f = abslx::bind_front(Print, std::ref(hi), "Guest"); // BAD!
//
// Example: Storing reference-like types.
//
//   void Print(abslx::string_view a, const std::string& b) {
//     std::cerr << a << b;
//   }
//
//   std::string hi = "Hello, ";
//   // Copies "hi".
//   abslx::bind_front(Print, hi)("Chuk");
//
//   // Compile error: std::reference_wrapper<const string> is not implicitly
//   // convertible to string_view.
//   // abslx::bind_front(Print, std::cref(hi))("Chuk");
//
//   // Doesn't copy "hi".
//   abslx::bind_front(Print, abslx::string_view(hi))("Chuk");
//
template <class F, class... BoundArgs>
constexpr functional_internal::bind_front_t<F, BoundArgs...> bind_front(
    F&& func, BoundArgs&&... args) {
  return functional_internal::bind_front_t<F, BoundArgs...>(
      abslx::in_place, abslx::forward<F>(func),
      abslx::forward<BoundArgs>(args)...);
}

ABSL_NAMESPACE_END
}  // namespace abslx

#endif  // ABSL_FUNCTIONAL_BIND_FRONT_H_
