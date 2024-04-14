#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:27:17 2022

@author: nakaishikai
"""
import random
from copy import copy
from typing import List

from chomsky_neural.data.utils import sample_from_list

random.seed(42)


def generate_AnBm(
    num_of_terms: int = 1,
    num_of_As: int = 10,
    num_of_Bs: int = 10,
    num_of_flip_A: int = 0,
    num_of_flip_B: int = 0,
    dist: str = "uniform",
) -> List[str]:
    """
    Generates AnBm.

    Arguments:
        num_of_terms {int} -- Number of terminals for each nonterminal.
        num_of_As {int} -- Number of As.
        num_of_Bs {int} -- Number of Bs.
        num_of_flip_A {int} -- Number of As to be flipped to generate wrong dependency.
        num_of_flip_B {int} -- Number of Bs to be flipped to generate wrong dependency.
        dist {str} -- Distribution of terminals. (default: {"uniform"})
    """
    term_A = list(range(0, num_of_terms, 1))
    term_B = list(range(num_of_terms, 2 * num_of_terms, 1))
    flip_sites_F = random.sample(list(range(0, num_of_As, 1)), num_of_flip_A)
    flip_sites_L = random.sample(list(range(0, num_of_Bs, 1)), num_of_flip_B)
    string: List[str] = []
    for i in range(0, num_of_As, 1):
        if i in flip_sites_F:
            string.append(
                str(sample_from_list(term_B, size=1, replace=True, dist=dist)[0])
            )
        else:
            string.append(
                str(sample_from_list(term_A, size=1, replace=True, dist=dist)[0])
            )
    for i in range(0, num_of_Bs, 1):
        if i in flip_sites_L:
            string.append(
                str(sample_from_list(term_A, size=1, replace=True, dist=dist)[0])
            )
        else:
            string.append(
                str(sample_from_list(term_B, size=1, replace=True, dist=dist)[0])
            )
    return string


def generate_AnBn(
    num_of_terms: int = 1,
    num_of_As: int = 10,
    num_of_flip_A: int = 0,
    num_of_flip_B: int = 0,
    dist: str = "uniform",
) -> List[str]:
    return generate_AnBm(
        num_of_terms, num_of_As, num_of_As, num_of_flip_A, num_of_flip_B, dist
    )


def generate_nested_dependency(
    num_of_nonterms: int,
    num_of_terms: int,
    num_of_dependencies: int,
    num_flip: int,
    dist: str = "uniform",
) -> List[str]:
    """
    Generates nested dependency.

    Arguments:
        num_of_nonterms {int} -- Number of nonterminals.
        num_of_terms {int} -- Number of terminals for each nonterminal.
        num_of_dependencies {int} -- Number of dependencies.
        num_flip {int} -- Number of nonterminals to be flipped to generate wrong dependency.
        dist {str} -- Distribution of terminals. (default: {"uniform"})
    """
    nonterms = list(range(0, num_of_nonterms, 1))
    string_n = []
    string_t = []
    for i in range(0, num_of_dependencies, 1):
        A = random.choice(nonterms)
        string_n.append(A)
        terms = list(range(A * num_of_terms, (A + 1) * num_of_terms, 1))
        string_t.append(
            str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
        )
    flip_sites = random.sample(list(range(0, num_of_dependencies, 1)), num_flip)
    for i in range(0, num_of_dependencies, 1):
        if i in flip_sites:
            nonterms_B = copy(nonterms)
            B = string_n[-(i + 1)]
            nonterms_B.remove(B)
            B_new = random.choice(nonterms_B)
            terms = list(
                range(
                    (num_of_nonterms + B_new) * num_of_terms,
                    (num_of_nonterms + B_new + 1) * num_of_terms,
                    1,
                )
            )
            string_t.append(
                str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
            )
        else:
            B = string_n[-(i + 1)]
            terms = list(
                range(
                    (num_of_nonterms + B) * num_of_terms,
                    (num_of_nonterms + B + 1) * num_of_terms,
                    1,
                )
            )
            string_t.append(
                str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
            )
    return string_t


def generate_cross_serial_dependency(
    num_of_nonterms: int,
    num_of_terms: int,
    num_of_dependencies: int,
    num_flip: int,
    dist: str = "uniform",
) -> List[str]:
    """
    Generates cross serial dependency.

    Arguments:
        num_of_nonterms {int} -- Number of nonterminals.
        num_of_terms {int} -- Number of terminals for each nonterminal.
        num_of_dependencies {int} -- Number of dependencies.
        num_flip {int} -- Number of nonterminals to be flipped to generate wrong dependency.
        dist {str} -- Distribution of terminals. (default: {"uniform"})
    """

    nonterms = list(range(0, num_of_nonterms, 1))
    string_n = []
    string_t = []
    for i in range(0, num_of_dependencies, 1):
        A = random.choice(nonterms)
        string_n.append(A)
        terms = list(range(A * num_of_terms, (A + 1) * num_of_terms, 1))
        string_t.append(
            str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
        )
    flip_sites = random.sample(list(range(0, num_of_dependencies, 1)), num_flip)
    for i in range(0, num_of_dependencies, 1):
        if i in flip_sites:
            nonterms_B = copy(nonterms)
            B = string_n[i]
            nonterms_B.remove(B)
            B_new = random.choice(nonterms_B)
            terms = list(
                range(
                    (num_of_nonterms + B_new) * num_of_terms,
                    (num_of_nonterms + B_new + 1) * num_of_terms,
                    1,
                )
            )
            string_t.append(
                str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
            )
        else:
            B = string_n[i]
            terms = list(
                range(
                    (num_of_nonterms + B) * num_of_terms,
                    (num_of_nonterms + B + 1) * num_of_terms,
                    1,
                )
            )
            string_t.append(
                str(sample_from_list(terms, size=1, replace=True, dist=dist)[0])
            )
    return string_t


# Archive
def AnBnCn(
    num_of_nonterms: int,
    num_of_terms: int,
    n: int,
    flip_A: int,
    flip_B: int,
    flip_C: int,
) -> List[str]:
    """
    Generates AnBnCn.

    Arguments:
        num_of_nonterms {int} -- Number of nonterminals.
        num_of_terms {int} -- Number of terminals for each nonterminal.
        n {int} -- Number of dependencies.
        flip_A {int} -- Number of nonterminals to be flipped to generate wrong dependency.
        flip_B {int} -- Number of nonterminals to be flipped to generate wrong dependency.
        flip_C {int} -- Number of nonterminals to be flipped to generate wrong dependency.

        caution: If flip_A = flip_B = flip_C \neq 0, it may happen that the correct sentence is generated by chance, so be careful.
    """
    nonterms = list(range(0, num_of_nonterms, 1))
    flip_sites_A = random.sample(list(range(0, n, 1)), flip_A)
    flip_sites_B = random.sample(list(range(0, n, 1)), flip_B)
    flip_sites_C = random.sample(list(range(0, n, 1)), flip_C)
    flip_sites = [flip_sites_A, flip_sites_B, flip_sites_C]
    string_n = []
    for i in range(0, n, 1):
        string_n.append(random.choice(nonterms))
    string_t: List[str] = []
    for f in range(0, 3, 1):
        for i in range(0, n, 1):
            if i in flip_sites[f]:
                nonterms_new = copy(nonterms)
                A = string_n[i]
                nonterms_new.remove(A)
                A_new = random.choice(nonterms_new)
                terms = list(
                    range(
                        (f * num_of_nonterms + A_new) * num_of_terms,
                        (f * num_of_nonterms + A_new + 1) * num_of_terms,
                        1,
                    )
                )
                string_t.append(str(random.choice(terms)))
            else:
                A = string_n[i]
                terms = list(
                    range(
                        (f * num_of_nonterms + A) * num_of_terms,
                        (f * num_of_nonterms + A + 1) * num_of_terms,
                        1,
                    )
                )
                string_t.append(str(random.choice(terms)))
    return string_t


def check_Dyck(string: List[int]) -> bool:
    val = 0
    for x in string:
        val = val + (1 - 2 * x)
        if val < 0:
            return False
    if val == 0:
        return True
    else:
        return False


def generate_balanced(n: int) -> List[int]:
    """
    Generates a balanced sentence of length 2n (a sentence with the same number of 0 and 1) uniformly.
    """
    site0 = random.sample(list(range(0, 2 * n, 1)), n)
    string = []
    for i in range(0, 2 * n, 1):
        if i in site0:
            string.append(0)
        else:
            string.append(1)
    return string


def gen_Dyck(n: int, max_itr: int) -> List[int]:
    # Generates a Dycke sentence of length 2n from a uniform distribution.
    # If a balanced sentence is generated randomly, it is output if it is Dyck.
    # If it is not found in max_itr times, an empty list is output.
    # The probability that a balanced sentence is Dyck is 1 / (n + 1), so it should be found in O (n) at most.

    for i in range(0, max_itr, 1):
        string = generate_balanced(n)
        if check_Dyck(string):
            return string
    return []


def Dyck_term(num_of_terms: int, n: int, max_itr: int) -> List[int]:
    # Dyck language + terminal．
    # num_of_terms: number of terminals for each nonterminal.
    string_n = gen_Dyck(n, max_itr)
    if len(string_n) == 0:
        return []
    else:
        string_t = []
        terms = [
            list(range(0, num_of_terms, 1)),
            list(range(num_of_terms, 2 * num_of_terms, 1)),
        ]
        for i in range(0, 2 * n, 1):
            a = random.choice(terms[string_n[i]])
            string_t.append(a)
        return string_t


def Dyck_term_negative_1(
    num_of_terms: int, n: int, flip_0: int, flip_1: int, max_itr: int
) -> List[int]:
    # Generates a negative instance of the Dyck language + terminal rule.
    # num_of_terms: The number of terminals corresponding to each nonterminal.
    # flip_0: Flip flip_0 of the n 0s to 1.
    # flip_1: Flip flip_1 of the n 1s to 0.
    # If flip_0 = flip_1 \neq 0, a correct sentence may be generated by chance, but it will be rejected if so.
    string_n = gen_Dyck(n, max_itr)
    if len(string_n) == 0:
        return []
    else:
        flip_sites_0 = random.sample(list(range(0, n, 1)), flip_0)
        flip_sites_1 = random.sample(list(range(0, n, 1)), flip_1)
        index_0 = 0
        index_1 = 0
        string_n_new = []
        for i in range(0, 2 * n, 1):
            if string_n[i] == 0:
                if index_0 in flip_sites_0:
                    string_n_new.append(1)
                else:
                    string_n_new.append(0)
                index_0 = index_0 + 1
            if string_n[i] == 1:
                if index_1 in flip_sites_1:
                    string_n_new.append(0)
                else:
                    string_n_new.append(1)
                index_1 = index_1 + 1
        if check_Dyck(string_n_new):
            return []
        else:
            print("original:", string_n)
            print("flipped:", string_n_new)
            string_t = []
            terms = [
                list(range(0, num_of_terms, 1)),
                list(range(num_of_terms, 2 * num_of_terms, 1)),
            ]
            for i in range(0, 2 * n, 1):
                a = random.choice(terms[string_n_new[i]])
                string_t.append(a)
            return string_t


def Dyck_term_negative_2(
    num_of_terms: int, n: int, flip: int, max_itr: int
) -> List[int]:
    # Generates a negative instance of the Dyck language + terminal rule.
    # num_of_terms: The number of terminals corresponding to each nonterminal.
    # flip: Flip flip of the 2n nonterminals.
    # If flip is even, a correct sentence may be generated by chance, but it will be rejected if so.
    string_n = gen_Dyck(n, max_itr)
    if len(string_n) == 0:
        return []
    else:
        flip_sites = random.sample(list(range(0, 2 * n, 1)), flip)
        string_n_new = []
        for i in range(0, 2 * n, 1):
            if i in flip_sites:
                string_n_new.append(1 - string_n[i])
            else:
                string_n_new.append(string_n[i])
        if check_Dyck(string_n_new):
            return []
        else:
            print("original:", string_n)
            print("flipped:", string_n_new)
            string_t = []
            terms = [
                list(range(0, num_of_terms, 1)),
                list(range(num_of_terms, 2 * num_of_terms, 1)),
            ]
            for i in range(0, 2 * n, 1):
                a = random.choice(terms[string_n_new[i]])
                string_t.append(a)
            return string_t
