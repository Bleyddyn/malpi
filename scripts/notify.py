#! /usr/bin/python
""" Use MaLPi's notify function from a command line script.
"""

import sys
import argparse

from malpi.notify import notify, get_email_config

def _parse_cmdline():
    parser = argparse.ArgumentParser(description='Send a notification by email and/or Mac OS X notification system .', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( 'title', nargs=1, help="Title for the notification and/or Subject for the email" );
    parser.add_argument( '--message', help="Message body. If none, expect body on standard input" );
    parser.add_argument( '--sub', help="Subtitle." );
    parser.add_argument( '--email', help="Email address to send the notification to");
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mac', action="store_true", default=True, help='Send a Mac OS X Notification')
    group.add_argument('--no_mac', action="store_false", dest='mac', default=False, help="Don't Send a Mac OS X Notification")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _parse_cmdline()

    if args.message:
        msg = args.message
    else:
        msg = sys.stdin.read()

    conf = get_email_config()
    notify( args.title[0], subTitle=args.sub, message=msg, email_to=args.email, mac=args.mac, email_config=conf )
