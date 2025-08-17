'use client';

import React, { useMemo, useState } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Card, PageHeader, Button, Badge } from '@/ui/core/components';
import UnauthorizedPage from '@/components/UnauthorizedPage';
import LoadingSpinner from '@/components/LoadingSpinner';
import { EyeIcon, ChatBubbleLeftIcon } from '@heroicons/react/24/outline';

type Trainee = {
  uid: string;
  name: string;
  email: string;
  status: 'active' | 'pending' | 'inactive';
  progressPct: number;
  lastActivity: string;
};

export default function TrainerDashboardPage(): JSX.Element {
  const { role } = useAuth();
  const [loading] = useState<boolean>(false);

  // Mock stats
  const activeTrainees = 12;
  const pendingRequests = 3;
  const recentActivities = '8 this week';

  // Mock trainee roster
  const trainees: Trainee[] = useMemo(
    () => [
      {
        uid: 'uid-1',
        name: 'Alice Johnson',
        email: 'alice@example.com',
        status: 'active',
        progressPct: 82,
        lastActivity: '2025-03-01T10:12:00Z',
      },
      {
        uid: 'uid-2',
        name: 'Bob Smith',
        email: 'bob@example.com',
        status: 'pending',
        progressPct: 14,
        lastActivity: '2025-02-25T08:40:00Z',
      },
      {
        uid: 'uid-3',
        name: 'Carla Gomez',
        email: 'carla@example.com',
        status: 'active',
        progressPct: 58,
        lastActivity: '2025-02-28T15:30:00Z',
      },
    ],
    []
  );

  // Profile completeness (mock)
  const profileCompleteness = 75;

  // Role guard: only trainers can view this dashboard
  if (!role) {
    // If role is not yet loaded, show a spinner
    return (
      <div className="flex items-center justify-center min-h-[40vh]">
        <LoadingSpinner />
      </div>
    );
  }

  if (role !== 'trainer') {
    return <UnauthorizedPage />;
  }

  return (
    <main className="max-w-6xl mx-auto p-8 app-px space-y-12">
      <PageHeader
        title="Trainer Dashboard"
        subtitle="Overview of your trainees, requests and recent activity."
      />

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-neutral-500">Active trainees</p>
              <p className="mt-2 text-2xl font-semibold text-white">{activeTrainees}</p>
            </div>
            <div className="text-right">
              <Badge className="bg-green-500 text-white">Active</Badge>
            </div>
          </div>
          <p className="mt-4 text-sm text-neutral-400">Trainees currently assigned to you</p>
        </Card>

        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-neutral-500">Pending requests</p>
              <p className="mt-2 text-2xl font-semibold text-white">{pendingRequests}</p>
            </div>
            <div className="text-right">
              <Badge className="bg-yellow-500 text-white">Pending</Badge>
            </div>
          </div>
          <p className="mt-4 text-sm text-neutral-400">New trainee connection requests</p>
        </Card>

        <Card>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-neutral-500">Recent activities</p>
              <p className="mt-2 text-2xl font-semibold text-white">{recentActivities}</p>
            </div>
            <div className="text-right">
              <Badge className="bg-blue-500 text-white">Activity</Badge>
            </div>
          </div>
          <p className="mt-4 text-sm text-neutral-400">Key events from your trainees</p>
        </Card>
      </div>

      {/* Trainee Roster */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">Trainee Roster</h3>
          <div className="flex items-center gap-3">
            <Button variant="primary" onClick={() => window.location.reload()}>
              Refresh
            </Button>
            <Button variant="ghost" onClick={() => {}}>
              Export
            </Button>
          </div>
        </div>

        <Card>
          {loading ? (
            <div className="py-8 flex justify-center">
              <LoadingSpinner />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full table-auto text-sm">
                <thead>
                  <tr className="text-left text-neutral-400 border-b border-border pb-2">
                    <th className="py-2 px-3">Name</th>
                    <th className="py-2 px-3">Email</th>
                    <th className="py-2 px-3">Status</th>
                    <th className="py-2 px-3">Progress</th>
                    <th className="py-2 px-3">Last Activity</th>
                    <th className="py-2 px-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {trainees.map((t) => (
                    <tr key={t.uid} className="border-b border-border">
                      <td className="py-3 px-3">{t.name}</td>
                      <td className="py-3 px-3 text-neutral-400">{t.email}</td>
                      <td className="py-3 px-3">
                        {t.status === 'active' ? (
                          <Badge className="bg-green-500 text-white">Active</Badge>
                        ) : t.status === 'pending' ? (
                          <Badge className="bg-yellow-500 text-white">Pending</Badge>
                        ) : (
                          <Badge className="bg-gray-600 text-white">Inactive</Badge>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <div className="w-[160px]">
                          <div className="bg-neutral-700 rounded-full h-2 overflow-hidden">
                            <div
                              className="h-2 rounded-full bg-gradient-to-r from-blue-400 to-blue-600"
                              style={{ width: `${t.progressPct}%` }}
                            />
                          </div>
                          <div className="text-xs text-neutral-400 mt-1">{t.progressPct}%</div>
                        </div>
                      </td>
                      <td className="py-3 px-3 text-neutral-400">
                        {new Date(t.lastActivity).toLocaleString()}
                      </td>
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            aria-label={`View ${t.name}`}
                            onClick={() => {
                              // Implement navigation to trainee profile
                              window.location.href = `/trainer/trainees/${t.uid}`;
                            }}
                          >
                            <EyeIcon className="w-4 h-4 mr-2" /> View
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            aria-label={`Message ${t.name}`}
                            onClick={() => {
                              window.location.href = `/messages/compose?to=${encodeURIComponent(
                                t.email
                              )}`;
                            }}
                          >
                            <ChatBubbleLeftIcon className="w-4 h-4 mr-2" /> Message
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {trainees.length === 0 && (
                <div className="p-6 text-center text-neutral-400">No trainees found</div>
              )}
            </div>
          )}
        </Card>
      </section>

      {/* Profile Completeness */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <h4 className="text-sm text-neutral-400">Profile completeness</h4>
          <div className="mt-3">
            <div className="flex items-center justify-between">
              <div className="text-lg font-medium">Profile {profileCompleteness}% Complete</div>
              <div className="text-sm text-neutral-400">{100 - profileCompleteness}% to go</div>
            </div>
            <div className="mt-3">
              <div className="w-full bg-neutral-700 rounded-full h-3 overflow-hidden">
                <div
                  className="h-3 rounded-full bg-gradient-to-r from-green-400 to-green-600"
                  style={{ width: `${profileCompleteness}%` }}
                />
              </div>
            </div>
            <div className="mt-4">
              <Button
                variant="primary"
                onClick={() => {
                  window.location.href = '/profile';
                }}
              >
                Complete Profile
              </Button>
            </div>
          </div>
        </Card>

        <Card>
          <h4 className="text-sm text-neutral-400">Coaching Tools</h4>
          <ul className="mt-3 space-y-2 text-sm text-neutral-300">
            <li>- Create workout plans</li>
            <li>- Send progress checks</li>
            <li>- Review trainee progress</li>
          </ul>
        </Card>

        <Card>
          <h4 className="text-sm text-neutral-400">Quick Actions</h4>
          <div className="mt-3 flex flex-col gap-3">
            <Button variant="ghost" onClick={() => {}}>
              Create Program
            </Button>
            <Button variant="ghost" onClick={() => {}}>
              Invite Trainee
            </Button>
          </div>
        </Card>
      </section>
    </main>
  );
}