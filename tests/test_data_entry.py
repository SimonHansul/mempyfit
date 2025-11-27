def test_constructing_minimal_dataset_from_scratch():
    # Create an empty dataset
    data = Dataset()

    # Add some hypothetical data
    data.add(
        name="L_m",
        value=0.4,
        units="cm",
        labels="maximum body length",
        temperature=C2K(20)
    )

    # Ensure dataset is of correct type
    assert isinstance(data, Dataset)

    # Ensure we can retrieve the value by name
    assert data["L_m"] == 0.4

    # Add more data
    data.add(
        name="cum_repro",
        value=60,
        units="#",
        labels="cumulative reproduction",
        temperature=C2K(20),
        comment="refers to cum. repro at the end of test (21d)"
    )

    # Check comments
    assert data.comments == ["", "refers to cum. repro at the end of test (21d)"]

    # Optional sanity check: dataset contains 2 entries
    assert len(data.names) == 2

    data.add(
        name="tL",
        value=np.array([
            [0, 0.1], 
            [2, 0.12], 
            [5, 0.2], 
            [7, 0.25], 
            [9, 0.26], 
            [12, 0.27], 
            [14, 0.27]
        ]),
        units=["d", "mm"],
        labels=["length growth over time"],
        temperature=C2K(20)
    )