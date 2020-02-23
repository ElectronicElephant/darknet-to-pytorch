def parse_cfg(fp):
    """
    Parses the darknet config file

    :return a list of dicts, each of which represents a block
    """

    with open(fp) as cfg:
        lines = [x.strip() for x in cfg.readlines()]  # Filter out all '\n's
        lines = [x for x in lines if len(x) > 0 and x[0] != '#']  # Filter out the comments and blank lines
        # print(lines)
        assert lines[0][0] == '['

        blocks = []
        block = {}
        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                block = {'type': line[1:-1]}
            else:
                key_, value_ = line.split('=')
                block[key_] = value_

        return blocks


if __name__ == '__main__':
    tmp = parse_cfg('./../src/yolov3.cfg')
    print(tmp)
